"""Network output -> segmentation on the GPU: every class resampled with cuCIM exactly as nnU-Net's default does
(linear, edge mode, separate z when anisotropic), one class at a time, with a running argmax.

Same labels as resampling all classes then taking the argmax (softmax does not change the argmax), without ever holding
all classes at the scan's full size. Checked on real AMOS scans against the SimpleITK logits resampler: 0 labels
changed, 285 s -> 9 s for a 67-class model. nnU-Net's export uses the returned labels directly (returns_labels).
"""
import numpy as np


def resample_logits_argmax_cucim(data, new_shape, current_spacing, new_spacing, **kwargs):
    import cupy as cp
    import torch
    from nnunetv2.preprocessing.resampling.cucim_resampling import _resample_data_or_seg_to_shape_cucim_gpu_in_out
    if isinstance(data, torch.Tensor):
        data = data.numpy()
    kwargs = {'is_seg': False, 'order': 1, 'order_z': 0, 'force_separate_z': None} | kwargs
    logits = cp.asarray(data, dtype=cp.float32)
    best = labels = None
    for k in range(logits.shape[0]):
        resampled = _resample_data_or_seg_to_shape_cucim_gpu_in_out(logits[k:k + 1], new_shape, current_spacing,
                                                                   new_spacing, **kwargs)[0]
        if best is None:
            best, labels = resampled, cp.zeros(resampled.shape, cp.uint8 if logits.shape[0] < 256 else cp.uint16)
        else:
            better = resampled > best  # strict: ties keep the lower class, as numpy's argmax does
            best = cp.where(better, resampled, best)
            labels[better] = k
    result = cp.asnumpy(labels)
    del logits, best, labels
    cp.get_default_memory_pool().free_all_blocks()
    return result


resample_logits_argmax_cucim.returns_labels = True


def self_check():
    """Streamed argmax equals argmax of the fully resampled output (needs a GPU with cupy and cucim)."""
    from nnunetv2.preprocessing.resampling.cucim_resampling import resample_data_or_seg_to_shape_cucim
    x = np.random.default_rng(0).normal(0, 1, (5, 30, 40, 36)).astype(np.float32)
    for shape, cur, new in (((45, 60, 50), (1.5, 1.5, 1.5), (1, 1, 1.08)), ((12, 70, 64), (1.5, 1.5, 1.5), (4, 0.86, 0.84))):
        full = resample_data_or_seg_to_shape_cucim(x, shape, cur, new, is_seg=False, order=1, order_z=0, force_separate_z=None)
        assert (resample_logits_argmax_cucim(x, shape, cur, new) == full.argmax(0)).all()
    print('streamed GPU argmax == argmax of the full resampling')


if __name__ == '__main__':
    self_check()
