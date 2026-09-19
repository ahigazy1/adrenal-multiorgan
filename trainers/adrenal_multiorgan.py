"""The one trainer used for the full run. Everything not overridden here is stock nnU-Net.

Differences from stock nnUNetTrainer, each chosen in PLAN.md:
  - batch size 4, 2000 epochs
  - fine-tuning schedule from nnU-Net's own PretrainedTrainer: SGD, peak LR 1e-3,
    50 epochs of linear warmup, then polynomial decay (exponent 0.9)
  - no mirroring, in training or inference
  - reduced augmentation: no blur, noise or simulated low resolution; mild rotation and scaling
  - the Dice term of the loss is replaced by focal Tversky (false negatives 0.6, false positives 0.4);
    cross-entropy and deep supervision are unchanged
  - a checkpoint every 100 epochs, written atomically and uploaded to Hugging Face together with
    the best checkpoint and the logs
"""
import os
from copy import deepcopy
from math import pi
from pathlib import Path

import torch
from torch import nn
from torch.optim.lr_scheduler import LambdaLR

from batchgeneratorsv2.transforms.intensity.gaussian_noise import GaussianNoiseTransform
from batchgeneratorsv2.transforms.noise.gaussian_blur import GaussianBlurTransform
from batchgeneratorsv2.transforms.spatial.low_resolution import SimulateLowResolutionTransform
from batchgeneratorsv2.transforms.spatial.spatial import SpatialTransform
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

BATCH_SIZE = 4
EPOCHS = 2000
WARMUP_EPOCHS = 50
PEAK_LR = 1e-3
CHECKPOINT_EVERY = 100
FALSE_POSITIVE_WEIGHT, FALSE_NEGATIVE_WEIGHT, FOCAL_EXPONENT, SMOOTH = 0.4, 0.6, 0.75, 1e-5


def learning_rate_factor(epoch):
    """Multiplier on PEAK_LR for a given epoch (0-based): linear warmup, then polynomial decay."""
    if epoch < WARMUP_EPOCHS:
        return (epoch + 1) / WARMUP_EPOCHS
    return max(0., 1 - (epoch - WARMUP_EPOCHS) / (EPOCHS - WARMUP_EPOCHS)) ** 0.9


class FocalTverskyLoss(nn.Module):
    """mean over foreground classes of (1 - TP / (TP + 0.4 FP + 0.6 FN)) ** 0.75, on softmax probabilities."""

    def __init__(self, batch_dice):
        super().__init__()
        self.batch_dice = batch_dice

    def forward(self, logits, target, loss_mask=None):
        # Sums are done in float32 even when the network runs in float16.
        probabilities = logits.float().softmax(1)[:, 1:]
        with torch.no_grad():
            truth = torch.zeros_like(logits, dtype=torch.bool).scatter_(1, target.long(), True)[:, 1:]
        if loss_mask is not None:
            probabilities = probabilities * loss_mask
            truth = truth & loss_mask.bool()
        axes = tuple(range(2, logits.ndim))
        if self.batch_dice:
            axes = (0,) + axes
        true_positive = (probabilities * truth).sum(axes, dtype=torch.float32)
        false_positive = (probabilities * ~truth).sum(axes, dtype=torch.float32)
        false_negative = ((1 - probabilities) * truth).sum(axes, dtype=torch.float32)
        error = FALSE_POSITIVE_WEIGHT * false_positive + FALSE_NEGATIVE_WEIGHT * false_negative
        # The 1e-7 keeps the gradient of x ** 0.75 finite when a class is predicted perfectly.
        return (error / (true_positive + error + SMOOTH) + 1e-7).pow(FOCAL_EXPONENT).mean()


class AdrenalMultiorgan(nnUNetTrainer):
    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device('cuda')):
        plans = deepcopy(plans)
        plans['configurations'][configuration]['batch_size'] = BATCH_SIZE
        super().__init__(plans, configuration, fold, dataset_json, device=device)
        self.num_epochs = EPOCHS
        self.initial_lr = PEAK_LR
        self.save_every = CHECKPOINT_EVERY

    def configure_optimizers(self):
        optimizer, _ = super().configure_optimizers()
        # nnU-Net calls scheduler.step(epoch) with the absolute epoch, so this also holds after a resume.
        return optimizer, LambdaLR(optimizer, learning_rate_factor)

    def configure_rotation_dummyDA_mirroring_and_inital_patch_size(self):
        rotation, dummy_2d, initial_patch_size, _ = super().configure_rotation_dummyDA_mirroring_and_inital_patch_size()
        self.inference_allowed_mirroring_axes = None
        return rotation, dummy_2d, initial_patch_size, None

    @staticmethod
    def get_training_transforms(*args, **kwargs):
        pipeline = nnUNetTrainer.get_training_transforms(*args, **kwargs)
        removed = (GaussianNoiseTransform, GaussianBlurTransform, SimulateLowResolutionTransform)
        pipeline.transforms = [t for t in pipeline.transforms if not isinstance(getattr(t, 'transform', t), removed)]
        spatial = next(t for t in pipeline.transforms if isinstance(t, SpatialTransform))
        spatial.rotation = (-10 * pi / 180, 10 * pi / 180)
        spatial.p_rotation = 0.1
        spatial.scaling = (0.9, 1.1)
        spatial.p_scaling = 0.1
        return pipeline

    def _build_loss(self):
        loss = super()._build_loss()  # deep supervision around (Dice + cross-entropy)
        focal = FocalTverskyLoss(self.configuration_manager.batch_dice)
        loss.loss.dc = torch.compile(focal) if self._do_i_compile() else focal
        return loss

    def save_checkpoint(self, filename):
        # Write to a temporary name first: a VM stopped mid-save must not leave a broken checkpoint.
        super().save_checkpoint(filename + '.tmp')
        if os.path.exists(filename + '.tmp'):
            os.replace(filename + '.tmp', filename)

    def on_epoch_end(self):
        finished_epochs = self.current_epoch + 1
        super().on_epoch_end()
        if finished_epochs % CHECKPOINT_EVERY == 0 and self.local_rank == 0 and not self.disable_checkpointing:
            upload(Path(self.output_folder).parent, f'epoch {finished_epochs}')


def upload(model_folder, message):
    """Upload checkpoints (latest, best, final), logs and settings. Unchanged files are not sent again."""
    from huggingface_hub import HfApi
    try:
        HfApi().create_repo(os.environ['ADRENAL_HF_REPO'], private=True, exist_ok=True)
        url = HfApi().upload_folder(repo_id=os.environ['ADRENAL_HF_REPO'], folder_path=model_folder,
                                    path_in_repo=model_folder.name, ignore_patterns=['*.tmp', '*.lock'],
                                    commit_message=f'{model_folder.name}: {message}')
        print(f'Uploaded to Hugging Face ({message}): {url}', flush=True)
    except Exception as error:
        # The checkpoint is safe on disk and the next upload includes it; an outage must not stop training.
        print(f'WARNING: Hugging Face upload failed ({message}), training continues: {error!r}', flush=True)
