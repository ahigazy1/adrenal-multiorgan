"""nnU-Net's predictor, but the sliding window sends several patches through the network at once.

Everything else is nnU-Net's: the same patches, tile step, Gaussian weighting and half-precision sum on the GPU.
The networks normalise each sample on its own (instance normalisation), so a patch's output does not depend on
which other patches share its batch. Run this file to check that on a small network: python batched_predictor.py
"""
import numpy as np
import torch
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.inference.sliding_window_prediction import compute_gaussian

VOXELS_PER_FORWARD = 19_000_000  # two of our 256x128x192 patches; smaller patches give larger batches


class BatchedPredictor(nnUNetPredictor):
    def _internal_predict_sliding_window_return_logits(self, data, slicers, do_on_device=True):
        if not do_on_device:  # nnU-Net's fallback when the GPU memory is too small: leave it as it is
            return super()._internal_predict_sliding_window_return_logits(data, slicers, do_on_device)
        patch = tuple(self.configuration_manager.patch_size)
        size = max(1, VOXELS_PER_FORWARD // int(np.prod(patch)))
        data = data.to(self.device)
        logits = torch.zeros((self.label_manager.num_segmentation_heads, *data.shape[1:]), dtype=torch.half, device=self.device)
        weights = torch.zeros(data.shape[1:], dtype=torch.half, device=self.device)
        gaussian = compute_gaussian(patch, sigma_scale=1. / 8, value_scaling_factor=10, device=self.device) if self.use_gaussian else 1
        for start in range(0, len(slicers), size):
            chunk = slicers[start:start + size]
            # the last batch is filled up with copies so that torch.compile sees one input shape only
            batch = torch.stack([data[s] for s in chunk + [chunk[-1]] * (size - len(chunk))])
            for prediction, s in zip(self._internal_maybe_mirror_and_predict(batch), chunk):
                logits[s] += prediction * gaussian
                weights[s[1:]] += gaussian
        torch.div(logits, weights, out=logits)
        if torch.any(torch.isinf(logits)):
            raise RuntimeError('Encountered inf in the predicted logits')
        return logits


def self_check():
    """Batched and nnU-Net's own sliding window give the same labels on a small instance-norm network."""
    from types import SimpleNamespace
    torch.manual_seed(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    network = torch.nn.Sequential(torch.nn.Conv3d(1, 8, 3, padding=1), torch.nn.InstanceNorm3d(8, affine=True),
                                  torch.nn.LeakyReLU(), torch.nn.Conv3d(8, 4, 3, padding=1)).to(device).eval()
    image = torch.randn(1, 40, 50, 70)
    results = []
    for kind in (nnUNetPredictor, BatchedPredictor):
        predictor = kind(tile_step_size=0.5, use_mirroring=False, perform_everything_on_device=True, device=device,
                         allow_tqdm=False)
        predictor.network = network
        predictor.configuration_manager = SimpleNamespace(patch_size=[32, 32, 32])
        predictor.label_manager = SimpleNamespace(num_segmentation_heads=4)
        results.append(predictor.predict_sliding_window_return_logits(image).float().cpu())
    difference = (results[0] - results[1]).abs().max().item()
    changed = (results[0].argmax(0) != results[1].argmax(0)).sum().item()
    print(f'{device}: largest logit difference {difference:.2e}, labels changed {changed} of {results[0][0].numel()}')
    assert difference < 1e-2 and changed <= results[0][0].numel() * 1e-4


if __name__ == '__main__':
    self_check()
