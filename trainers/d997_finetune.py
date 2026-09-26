"""Trainer for the D997 fine-tune (Dataset903, ADRENAL_RUN=d997). AdrenalMultiorgan's loss (CE + focal Tversky),
fine-tuning schedule shape and atomic checkpoints, with these differences:
  - D997's own batch size (2) and patch size from its plans
  - stock nnU-Net augmentation, still without mirroring (flipping would swap left and right structures)
  - EPOCHS epochs; a local checkpoint every 25 epochs as the resume point after a Spot stop
  - Hugging Face snapshots every 250 epochs, not 100: each one stays in the repository's history
AdrenalMultiorgan itself is not edited: its file hash is part of the finished Dataset902 run's record.
"""
from pathlib import Path

import torch
from torch.optim.lr_scheduler import LambdaLR

from adrenal_multiorgan import PEAK_LR, WARMUP_EPOCHS, AdrenalMultiorgan, upload
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.nnUNetTrainer.variants.data_augmentation.nnUNetTrainerNoMirroring import nnUNetTrainerNoMirroring

EPOCHS = 1000  # ponytail: fixed before launch from the smoke test's seconds per epoch
UPLOAD_EVERY = 250


def learning_rate_factor(epoch):
    """Linear warmup, then polynomial decay (exponent 0.9), as AdrenalMultiorgan but over EPOCHS."""
    if epoch < WARMUP_EPOCHS:
        return (epoch + 1) / WARMUP_EPOCHS
    return max(0., 1 - (epoch - WARMUP_EPOCHS) / (EPOCHS - WARMUP_EPOCHS)) ** 0.9


class D997Finetune(AdrenalMultiorgan):
    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device('cuda')):
        # Skips AdrenalMultiorgan.__init__, which would force its batch size of 4 onto D997's plans.
        nnUNetTrainerNoMirroring.__init__(self, plans, configuration, fold, dataset_json, device=device)
        self.num_epochs = EPOCHS
        self.initial_lr = PEAK_LR
        self.save_every = 25

    def configure_optimizers(self):
        optimizer, _ = nnUNetTrainer.configure_optimizers(self)
        return optimizer, LambdaLR(optimizer, learning_rate_factor)

    get_training_transforms = staticmethod(nnUNetTrainer.get_training_transforms)  # stock; mirror_axes is None here

    def on_epoch_end(self):
        finished_epochs = self.current_epoch + 1
        nnUNetTrainerNoMirroring.on_epoch_end(self)
        if finished_epochs % UPLOAD_EVERY == 0 and self.local_rank == 0 and not self.disable_checkpointing:
            upload(Path(self.output_folder).parent, f'epoch {finished_epochs}')


if __name__ == '__main__':
    assert learning_rate_factor(0) == 1 / WARMUP_EPOCHS and learning_rate_factor(WARMUP_EPOCHS - 1) == 1
    assert learning_rate_factor(EPOCHS) == 0 and 0 < learning_rate_factor(EPOCHS // 2) < 1
