"""CPU check of trainers/adrenal_multiorgan.py: no GPU, no data, a few seconds.

    python check_trainer.py path/to/plans.json

Confirms the schedule, batch size, mirroring, augmentation and loss are what PLAN.md says.
"""
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
scratch = tempfile.mkdtemp()
for name in ('nnUNet_raw', 'nnUNet_preprocessed', 'nnUNet_results'):
    os.environ[name] = scratch
os.environ['nnUNet_compile'] = 'false'
sys.path.insert(0, str(ROOT / 'trainers'))

import torch
from adrenal_multiorgan import AdrenalMultiorgan, FocalTverskyLoss, learning_rate_factor, PEAK_LR
from batchgeneratorsv2.transforms.intensity.gaussian_noise import GaussianNoiseTransform
from batchgeneratorsv2.transforms.noise.gaussian_blur import GaussianBlurTransform
from batchgeneratorsv2.transforms.spatial.low_resolution import SimulateLowResolutionTransform
from batchgeneratorsv2.transforms.spatial.mirroring import MirrorTransform

LABELS = ['background', 'adrenal_left', 'adrenal_right', 'kidney_left', 'kidney_right', 'liver', 'spleen', 'aorta',
          'inferior_vena_cava', 'pancreas']

# Schedule: 0.00002 at the first epoch, the peak at epochs 50 and 51, decaying to nearly zero at the end.
rates = [PEAK_LR * learning_rate_factor(epoch) for epoch in range(2000)]
assert abs(rates[0] - 2e-5) < 1e-12 and abs(rates[49] - 1e-3) < 1e-12 and abs(rates[50] - 1e-3) < 1e-12
assert all(a > b for a, b in zip(rates[50:], rates[51:])) and 0 < rates[-1] < 2e-6

# Loss against the formula written out by hand, in float64.
torch.manual_seed(0)
logits = torch.randn(2, 10, 4, 5, 6, requires_grad=True)
target = torch.randint(0, 10, (2, 1, 4, 5, 6))
for batch_dice in (False, True):
    p, terms = logits.double().softmax(1), []
    for b in ([slice(None)] if batch_dice else range(2)):
        for c in range(1, 10):
            prob, truth = p[b, c], (target[b, 0] == c).double()
            tp, fp, fn = (prob * truth).sum(), (prob * (1 - truth)).sum(), ((1 - prob) * truth).sum()
            terms.append((1 - (tp + 1e-5) / (tp + 0.4 * fp + 0.6 * fn + 1e-5)) ** 0.75)
    value = FocalTverskyLoss(batch_dice)(logits, target)
    torch.testing.assert_close(value.double(), torch.stack(terms).mean(), rtol=1e-5, atol=1e-7)
    gradient, = torch.autograd.grad(value, logits)
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
half = logits.detach().half().requires_grad_()  # float16 network output, float32 loss
FocalTverskyLoss(True)(half, target).backward()
assert torch.isfinite(half.grad).all()

# The trainer itself, built exactly as nnU-Net builds it.
plans = json.loads(Path(sys.argv[1]).read_text())
plans['continue_training'] = False  # nnU-Net's run_training sets this key before building a trainer
dataset = {'channel_names': {'0': 'CT'}, 'labels': {name: i for i, name in enumerate(LABELS)},
           'numTraining': 1, 'file_ending': '.nii.gz'}
trainer = AdrenalMultiorgan(plans, '3d_fullres', 0, dataset, device=torch.device('cpu'))
assert (trainer.num_epochs, trainer.save_every, trainer.initial_lr) == (2000, 100, 1e-3)
assert trainer.configuration_manager.batch_size == 4
assert trainer.configure_rotation_dummyDA_mirroring_and_inital_patch_size()[3] is None
assert trainer.inference_allowed_mirroring_axes is None
loss = trainer._build_loss()
assert isinstance(loss.loss.dc, FocalTverskyLoss) and loss.loss.weight_ce == loss.loss.weight_dice == 1
patch = trainer.configuration_manager.patch_size
transforms = trainer.get_training_transforms(patch, (-0.5, 0.5), None, None, do_dummy_2d_data_aug=False,
                                             foreground_labels=trainer.label_manager.foreground_labels).transforms
kinds = [type(getattr(t, 'transform', t)) for t in transforms]
assert not set(kinds) & {GaussianNoiseTransform, GaussianBlurTransform, SimulateLowResolutionTransform, MirrorTransform}
# Checkpoints: written under the final name only (no leftover .tmp), and an upload is attempted at epoch 100, not 99.
# (nnU-Net's own save and logging are replaced by stand-ins: building the real network needs the dataset.)
import adrenal_multiorgan
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
nnUNetTrainer.save_checkpoint = lambda self, filename: Path(filename).write_text('weights')
checkpoint = os.path.join(scratch, 'checkpoint_latest.pth')
trainer.save_checkpoint(checkpoint)
assert os.path.exists(checkpoint) and not os.path.exists(checkpoint + '.tmp')
uploads = []
adrenal_multiorgan.upload = lambda folder, message: uploads.append(message)
nnUNetTrainer.on_epoch_end = lambda self: setattr(self, 'current_epoch', self.current_epoch + 1)  # skip real logging
for trainer.current_epoch in (98, 99):
    trainer.on_epoch_end()
assert uploads == ['epoch 100'], uploads
print('Trainer check passed: schedule, batch size 4, no mirroring, reduced augmentation, focal Tversky 0.6/0.4, atomic checkpoints, upload every 100 epochs.')
