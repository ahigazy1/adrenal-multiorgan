"""SimpleITK backed resample_data_or_seg_to_shape for nnU-Net's plans system.

Drop in for nnunetv2.preprocessing.resampling.default_resampling.resample_data_or_seg_to_shape.
Signature identical so plans level functools.partial against either is
interchangeable. Internally constructs a per channel sitk.Image with
origin=(0,0,0), direction=identity, spacing=current_spacing, calls
sitk.Resample with multithreaded SITK (SetGlobalDefaultNumberOfThreads),
returns the resampled array.

This module is what gets called when a plans file's
resampling_fn_data/seg/probabilities entry points to
"resample_data_or_seg_to_shape_sitk".

Conventions.

nnU-Net passes data in shape (C, X, Y, Z) where X/Y/Z are abstract spatial
axes. SITK's GetImageFromArray takes a (z, y, x) array and produces an image
with GetSize() = (x, y, z). To map nnU-Net's (X, Y, Z) to sitk's (x, y, z)
consistently we reverse the spacing tuple before SetSpacing, reverse
new_shape before passing to sitk.Resample, and rely on the fact that the
reversal then reversal is self cancelling. The returned array shape matches
nnU-Net's expected (X', Y', Z').

Threading reads SITK_THREADS from env var, default 8.

For is_seg=True with order=0 we use sitkNearestNeighbor. With order>=1 we
use sitkLabelLinear (label aware per class linear). For is_seg=False with
order=3 sitkBSpline, order=1 sitkLinear, order=0 sitkNearestNeighbor.

Separate z anisotropic path is NOT implemented. SITK's resampler does not
natively distinguish in plane vs through plane interpolation. If a plans
file sets force_separate_z=True with this resampler the flag is ignored and
a single 3D resample is performed.
"""
from __future__ import annotations

import os
from typing import List, Tuple, Union

import numpy as np
import SimpleITK as sitk


__all__ = ["resample_data_or_seg_to_shape_sitk"]


_SITK_THREADS_SET = False


def _ensure_sitk_threads():
    global _SITK_THREADS_SET
    if _SITK_THREADS_SET:
        return
    n = int(os.environ.get("SITK_THREADS", "8"))
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(n)
    _SITK_THREADS_SET = True


def _interp_for(is_seg: bool, order: int):
    if is_seg:
        if order <= 0:
            return sitk.sitkNearestNeighbor
        return sitk.sitkLabelLinear
    if order >= 3:
        return sitk.sitkBSpline
    if order == 1:
        return sitk.sitkLinear
    return sitk.sitkNearestNeighbor


def _resample_one_channel(arr_xyz_nnunet: np.ndarray,
                          new_shape_xyz_nnunet: Tuple[int, ...],
                          current_spacing_xyz_nnunet: Tuple[float, ...],
                          new_spacing_xyz_nnunet: Tuple[float, ...],
                          interp) -> np.ndarray:
    # GetImageFromArray expects (z, y, x). nnU-Net's (X, Y, Z) reads as
    # (X=z_sitk, Y=y_sitk, Z=x_sitk). Reverse the spacing and shape tuples.
    img = sitk.GetImageFromArray(arr_xyz_nnunet)
    img.SetSpacing(tuple(float(s) for s in reversed(current_spacing_xyz_nnunet)))
    img.SetOrigin((0.0, 0.0, 0.0))
    img.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))

    new_size_sitk = [int(s) for s in reversed(new_shape_xyz_nnunet)]
    new_spacing_sitk = [float(s) for s in reversed(new_spacing_xyz_nnunet)]

    out = sitk.Resample(
        img, new_size_sitk, sitk.Transform(), interp,
        img.GetOrigin(), new_spacing_sitk, img.GetDirection(),
        0, img.GetPixelID(),
    )
    return sitk.GetArrayFromImage(out)


def resample_data_or_seg_to_shape_sitk(
    data,
    new_shape: Union[Tuple[int, ...], List[int], np.ndarray],
    current_spacing: Union[Tuple[float, ...], List[float], np.ndarray],
    new_spacing: Union[Tuple[float, ...], List[float], np.ndarray],
    is_seg: bool = False,
    order: int = 3,
    order_z: int = 0,
    force_separate_z: Union[bool, None] = False,
    separate_z_anisotropy_threshold: float = 3.0,
):
    """SITK backed drop in for resample_data_or_seg_to_shape. See module
    docstring for semantics and caveats."""
    _ensure_sitk_threads()

    try:
        import torch  # type: ignore
        if isinstance(data, torch.Tensor):
            data = data.numpy()
    except Exception:
        pass

    assert data is not None and data.ndim == 4, "data must be (c, x, y, z)"
    new_shape = tuple(int(s) for s in new_shape)
    in_shape = tuple(int(s) for s in data.shape[1:])
    if in_shape == new_shape:
        # No op (matches upstream)
        return data.astype(data.dtype if is_seg else np.float32, copy=False)

    interp = _interp_for(is_seg, order)
    out_dtype = data.dtype if is_seg else np.float32
    out = np.empty((data.shape[0], *new_shape), dtype=out_dtype)
    for c in range(data.shape[0]):
        # SITK GetImageFromArray works on contiguous numpy arrays.
        arr_in = np.ascontiguousarray(data[c]).astype(
            np.float32 if not is_seg else (np.int16 if data.dtype.kind in "iu" else data.dtype),
            copy=False,
        )
        resampled = _resample_one_channel(
            arr_in, new_shape, tuple(current_spacing), tuple(new_spacing), interp,
        )
        # sitk array shape = (z_sitk, y_sitk, x_sitk) = (X_nnunet, Y_nnunet, Z_nnunet)
        # because we reversed before SetSpacing. Already in nnU-Net order.
        out[c] = resampled.astype(out_dtype, copy=False)
    return out
