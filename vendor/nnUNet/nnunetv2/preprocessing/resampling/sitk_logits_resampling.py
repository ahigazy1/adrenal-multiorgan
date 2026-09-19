"""SimpleITK-native inverse/LOGITS resampler for nnU-Net (CPU, multithreaded, physical-grid).

Per-channel sitk.Resample of a (C,X,Y,Z) array to a target SHAPE, with a separate-z 2-stage for
anisotropic targets and a CONFIGURABLE interpolator (per-purpose order; not B-spline-3 for everything).
Drop-in for resampling_fn_probabilities (the decode of raw network logits before argmax); usable as
resampling_fn_data too. Registered purely by function name (recursive_find_resampling_fn_by_name walks
this dir) -- set the plan string + kwargs.

Physical-grid rationale: for a VOLUME phenotype, resampling in true physical space (spacing-aware) with
an explicit output Size reproduces nnU-Net's exact target shape while placing the boundary physically.

NOTE: is_seg is honored only for dtype; this fn does NOT do per-label seg (a spline on a multi-label map
blends ids) -- it is for DATA / LOGITS (is_seg=False). SimpleITK is imported lazily so the module is
import-safe on non-SITK envs (module discovery imports every file in this dir).
"""
from __future__ import annotations
import os
from typing import Union

import numpy as np

from nnunetv2.preprocessing.resampling.default_resampling import determine_do_sep_z_and_axis

_THREADS_SET = False


def _ensure_threads(sitk):
    global _THREADS_SET
    if not _THREADS_SET:
        sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(int(os.environ.get("SITK_THREADS", "8")))
        _THREADS_SET = True


def _interp_by_order(order, sitk):
    # Fail loud: do NOT silently promote an unsupported order (2/4/5) to cubic -- that would be a silent
    # interpolation-order change vs the scipy reference (scipy order 2 is quadratic, not B-spline-3).
    m = {0: sitk.sitkNearestNeighbor, 1: sitk.sitkLinear, 3: sitk.sitkBSpline}
    o = int(order)
    if o not in m:
        raise ValueError(f"unsupported resample order {o}; supported {sorted(m)} (0=NN, 1=linear, 3=cubic)")
    return m[o]


def _named_interp(name, sitk):
    # NOTE: deliberately NO 'gaussian' -- sitkGaussian is a smoothing kernel, NOT an interpolant (it smears a
    # delta across ~1000 voxels and would destroy small-gland logit mass). Windowed-sinc are the sharp options.
    return {"nearest": sitk.sitkNearestNeighbor, "linear": sitk.sitkLinear, "bspline": sitk.sitkBSpline,
            "lanczos": sitk.sitkLanczosWindowedSinc, "cosine": sitk.sitkCosineWindowedSinc,
            "hamming": sitk.sitkHammingWindowedSinc, "blackman": sitk.sitkBlackmanWindowedSinc}[name.lower()]


def _resample_full(arr_xyz, new_shape, interp, sitk):
    """One sitk.Resample reproducing nnU-Net's INDEX-space resize EXACTLY. nnU-Net's scipy/skimage path
    (default_resampling.resample_data_or_seg) maps output voxel o to input continuous index
    `scale*(o+0.5) - 0.5` with scale = N_in/N_out (align_corners=False, the half-pixel center convention --
    the code literally copies sklearn's resize) and edge-clamp boundary (mode='edge' / 'nearest').

    To match it physics-agnostically: input spacing 1, output spacing = scale, output origin = 0.5*(scale-1)
    -> sitk maps o -> origin + o*scale = scale*(o+0.5) - 0.5 (identical). NN-extrapolate out-of-range border
    samples == scipy mode='nearest'/'edge'. The passed spacings only set the SHAPE (new_shape is explicit),
    so the abstract zoom is exact regardless of the physical spacings. Reverse-tuple: numpy (X,Y,Z) -> sitk (Z,Y,X)."""
    arr_xyz = np.ascontiguousarray(arr_xyz, dtype=np.float32)
    in_shape = arr_xyz.shape
    scale = [float(in_shape[d]) / float(int(new_shape[d])) for d in range(3)]    # numpy-axis index ratio
    img = sitk.GetImageFromArray(arr_xyz)                                        # sitk size = reversed(in_shape)
    img.SetSpacing((1.0, 1.0, 1.0)); img.SetOrigin((0.0, 0.0, 0.0))
    img.SetDirection((1., 0., 0., 0., 1., 0., 0., 0., 1.))
    rs = sitk.ResampleImageFilter()
    rs.SetSize([int(new_shape[2]), int(new_shape[1]), int(new_shape[0])])        # sitk (Z,Y,X)
    rs.SetOutputSpacing((scale[2], scale[1], scale[0]))
    rs.SetOutputOrigin((0.5 * (scale[2] - 1.0), 0.5 * (scale[1] - 1.0), 0.5 * (scale[0] - 1.0)))
    rs.SetOutputDirection(img.GetDirection()); rs.SetTransform(sitk.Transform())
    rs.SetInterpolator(interp); rs.SetDefaultPixelValue(0.0)
    rs.SetUseNearestNeighborExtrapolator(True)                                   # == scipy mode='nearest'/'edge'
    return sitk.GetArrayFromImage(rs.Execute(img))                              # -> (X,Y,Z)


def _resample_sepz(arr_xyz, new_shape, axis, in_interp, z_interp, sitk):
    """2-stage separate-z: in-plane at in_interp HOLDING the anisotropic axis (scale=1 -> identity on that
    axis, so a 3D resample with one axis held == per-slice 2D resize), then the axis alone at z_interp.
    Matches nnU-Net's sep-z (in-plane skimage resize per slice + through-axis map_coordinates order_z)."""
    inter_shape = list(new_shape); inter_shape[axis] = int(arr_xyz.shape[axis])
    s1 = _resample_full(arr_xyz, inter_shape, in_interp, sitk)   # in-plane (axis = identity)
    return _resample_full(s1, new_shape, z_interp, sitk)          # through-axis only


def resample_logits_to_shape_sitk(data, new_shape, current_spacing, new_spacing,
                                  is_seg: bool = False, order: int = 1, order_z: int = 0,
                                  force_separate_z: Union[bool, None] = None,
                                  separate_z_anisotropy_threshold: float = 3.0,
                                  interpolator: Union[str, None] = None, **_):
    """SITK per-channel resample (C,X,Y,Z) -> (C,*new_shape). LOGITS: is_seg=False, order=1 (linear).
    `interpolator` (or env SITK_INTERP) overrides the order->interp map for exploration:
    linear|bspline|lanczos|cosine|hamming|blackman|nearest (no gaussian -- that's a blur, not an interpolant)."""
    import SimpleITK as sitk
    _ensure_threads(sitk)
    data = data.numpy() if hasattr(data, "numpy") else np.asarray(data)
    assert data.ndim == 4, f"expected (C,X,Y,Z), got {data.shape}"
    # Fail closed: this resampler is for DATA/LOGITS. A spline/linear pass over a multi-label seg blends ids;
    # only order-0 NN is id-safe. Route real seg resampling to sitk_resampling.py's label-aware path.
    assert (not is_seg) or int(order) == 0, \
        "sitk_logits resampler is DATA/LOGITS-only (is_seg=False); order>0 on a label map blends ids"
    new_shape = tuple(int(s) for s in new_shape)
    if tuple(data.shape[1:]) == new_shape:
        # copy=True: never hand back a view aliasing the caller's logit buffer. decode_compare reuses the
        # cached logits across every arm; an aliasing no-op return would let a later op corrupt other arms.
        return data.astype(data.dtype if is_seg else np.float32, copy=True)
    name = interpolator or os.environ.get("SITK_INTERP")
    in_interp = _named_interp(name, sitk) if name else _interp_by_order(order, sitk)
    z_interp = _interp_by_order(order_z, sitk)
    do_sep, axis = determine_do_sep_z_and_axis(force_separate_z, current_spacing, new_spacing,
                                               separate_z_anisotropy_threshold)
    out = np.zeros((data.shape[0], *new_shape), dtype=np.float32)
    for c in range(data.shape[0]):
        arr = np.asarray(data[c], dtype=np.float32)
        out[c] = (_resample_sepz(arr, new_shape, int(axis), in_interp, z_interp, sitk)
                  if do_sep else _resample_full(arr, new_shape, in_interp, sitk))
    return out


# ----------------------------------------------------------------------------------------------------------
# TRUE PHYSICAL-GRID resampler (the "physical grid for a volume phenotype" thesis). This does NOT match
# scipy's index grid. It samples in real physical/mm space with the ITK voxel-center convention:
#   output voxel o -> physical pos o*new_sp (origin 0) -> input continuous index o*new_sp/cur_sp = o*scale_phys.
# Differences from the index-matched `_resample_full` above: (1) scale = real new_sp/cur_sp (the spacing ratio)
# NOT N_in/N_out (the shape ratio) -- they differ sub-voxel by the shape-rounding; (2) NO half-pixel origin
# offset (voxel-center, o*scale) vs scipy's scale*(o+0.5)-0.5. Edge-clamp kept (UseNearestNeighborExtrapolator)
# so the ONLY variables vs scipy are the grid scale + the half-pixel convention, not the boundary fill.
def _resample_full_phys(arr_xyz, new_shape, cur_sp, new_sp, interp, sitk):
    arr_xyz = np.ascontiguousarray(arr_xyz, dtype=np.float32)
    img = sitk.GetImageFromArray(arr_xyz)                                        # sitk size = reversed(shape)
    img.SetSpacing(tuple(float(s) for s in cur_sp[::-1]))                        # REAL input spacing (sitk x,y,z)
    img.SetOrigin((0.0, 0.0, 0.0)); img.SetDirection((1., 0., 0., 0., 1., 0., 0., 0., 1.))
    rs = sitk.ResampleImageFilter()
    rs.SetSize([int(new_shape[2]), int(new_shape[1]), int(new_shape[0])])        # sitk (Z,Y,X)
    rs.SetOutputSpacing(tuple(float(s) for s in new_sp[::-1]))                   # REAL output spacing
    rs.SetOutputOrigin((0.0, 0.0, 0.0))                                          # voxel-center, NO half-pixel
    rs.SetOutputDirection(img.GetDirection()); rs.SetTransform(sitk.Transform())
    rs.SetInterpolator(interp); rs.SetDefaultPixelValue(0.0)
    rs.SetUseNearestNeighborExtrapolator(True)                                   # edge-clamp (fair vs scipy)
    return sitk.GetArrayFromImage(rs.Execute(img))


def _resample_sepz_phys(arr_xyz, new_shape, cur_sp, new_sp, axis, in_interp, z_interp, sitk):
    """Physical-grid 2-stage sep-z: hold the anisotropic axis (out spacing/size = input on that axis) for the
    in-plane pass, then resample the axis alone at z_interp -- same policy as the index sep-z, physical mapping."""
    inter_shape = list(new_shape); inter_shape[axis] = int(arr_xyz.shape[axis])
    inter_sp = list(new_sp); inter_sp[axis] = float(cur_sp[axis])
    s1 = _resample_full_phys(arr_xyz, inter_shape, cur_sp, inter_sp, in_interp, sitk)
    return _resample_full_phys(s1, new_shape, inter_sp, new_sp, z_interp, sitk)


def resample_data_to_shape_sitk_physical(data, new_shape, current_spacing, new_spacing,
                                         is_seg: bool = False, order: int = 3, order_z: int = 0,
                                         force_separate_z: Union[bool, None] = None,
                                         separate_z_anisotropy_threshold: float = 3.0,
                                         interpolator: Union[str, None] = None, **_):
    """TRUE physical-grid (mm-space, voxel-center) per-channel resample (C,X,Y,Z) -> (C,*new_shape). NOT
    index-matched to scipy -- this is the candidate for testing whether the physical grid changes a VOLUME
    phenotype. is_seg only guards order (a spline blends ids); intended for DATA (order 3)."""
    import SimpleITK as sitk
    _ensure_threads(sitk)
    data = data.numpy() if hasattr(data, "numpy") else np.asarray(data)
    assert data.ndim == 4, f"expected (C,X,Y,Z), got {data.shape}"
    assert (not is_seg) or int(order) == 0, "physical resampler is DATA-only (is_seg=False); order>0 blends ids"
    new_shape = tuple(int(s) for s in new_shape)
    if tuple(data.shape[1:]) == new_shape:
        return data.astype(np.float32, copy=True)
    name = interpolator or os.environ.get("SITK_INTERP")
    in_interp = _named_interp(name, sitk) if name else _interp_by_order(order, sitk)
    z_interp = _interp_by_order(order_z, sitk)
    do_sep, axis = determine_do_sep_z_and_axis(force_separate_z, current_spacing, new_spacing,
                                               separate_z_anisotropy_threshold)
    out = np.zeros((data.shape[0], *new_shape), dtype=np.float32)
    for c in range(data.shape[0]):
        arr = np.asarray(data[c], dtype=np.float32)
        out[c] = (_resample_sepz_phys(arr, new_shape, current_spacing, new_spacing, int(axis), in_interp, z_interp, sitk)
                  if do_sep else _resample_full_phys(arr, new_shape, current_spacing, new_spacing, in_interp, sitk))
    return out
