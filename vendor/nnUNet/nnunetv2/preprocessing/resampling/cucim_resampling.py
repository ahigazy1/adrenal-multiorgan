"""GPU accelerated cucim port of nnU-Net's resample_data_or_seg_to_shape.

This module sits at `nnunetv2.preprocessing.resampling.cucim_resampling`. The
three public functions `resample_data_or_seg_to_shape_cucim`,
`resample_seg_channelargmax_to_shape_cucim`, and
`resample_seg_volmatch_to_shape_cucim` are discoverable by
`recursive_find_resampling_fn_by_name` and can be referenced from a plans
JSON as drop in replacements for the upstream scipy resampler.

`resample_seg_volmatch_to_shape_cucim` is the volume preserving seg resampler:
each foreground class is resized as a float membership and thresholded at the
count matched isovalue (via `cp.partition`, device resident) so the class
physical volume is preserved across the resample. It removes the forward volume
drift the fixed 0.5 threshold leaves on small glands. Isotropic path only
(force_separate_z=False, the pinned adrenal configuration).

Semantics.

`is_seg=False, order=N` runs per channel `cucim.skimage.transform.resize`
with `mode='edge'`, `anti_aliasing=False`, `preserve_range=True`. Matches
`skimage.transform.resize` within float32 vs float64 noise around 1e-5.

`is_seg=True, order=N` runs per class binary resize plus a 0.5 threshold,
which mirrors `batchgenerators.augmentations.utils.resize_segmentation`.

Separate z anisotropic path uses 2D in plane resize plus 1D
`map_coordinates` along the low resolution axis with the exact
`scale * (idx + 0.5) - 0.5` formula nnU-Net uses.

`cp.unique(...)` returns sorted ascending, so class 0 (background) is
painted first and foreground classes win ties on the 0.5 threshold. This
matches upstream sorted iteration behavior.

Architecture, single primitive + two surfaces.

Every per channel kernel lives as a pure CuPy in / CuPy out primitive
(suffix `_gpu_in_out`). The NumPy entry points (no suffix) are thin
wrappers that upload, call the primitive, download. The single
implementation prevents drift between the host returning surface and the
device returning surface used by AdrenalGPUPreprocessor. Same pattern as
`_channelargmax_streaming_gpu` which already backs both
`resample_seg_channelargmax_to_shape_cucim` (host) and
`_resample_seg_channelargmax_gpu_out` (device).

The `_gpu_in_out` primitives wrap cuCIM kernel inputs with
`cp.ascontiguousarray(...)` after any dtype cast so transposed or sliced
CuPy views do not hit the silent perf cliff of strided cuCIM input.

Lazy imports.

cupy and cucim are imported inside each function body, NOT at module top.
This lets nnU-Net's `recursive_find_resampling_fn_by_name` walk the
preprocessing/resampling/ directory and import this module on a CPU only
machine without raising. The actual cupy and cucim imports only fire when
someone calls a function, which only happens when plans JSON selects this
backend.

Backend fallback.

Setting `NNUNET_CUCIM_BACKEND=cpu` at process start makes
`resample_data_or_seg_to_shape_cucim` delegate to the scipy reference at
`nnunetv2.preprocessing.resampling.default_resampling.resample_data_or_seg_to_shape`
without ever touching cupy. Useful for multi process inference on a machine
where CUDA contexts in every worker are undesirable. The dispatcher inside
`AdrenalGPUPreprocessor._dispatch_data_resample_gpu` checks the same env
var before routing to the device-returning orchestrator
`_resample_data_or_seg_to_shape_cucim_gpu_in_out`, so the escape hatch is
honored from both surfaces.
"""
from __future__ import annotations

import os
from typing import List, Tuple, Union

import numpy as np

# Use nnU-Net's exact anisotropy machinery so the do separate z decision is
# byte identical to the scipy reference.
from nnunetv2.preprocessing.resampling.default_resampling import (
    determine_do_sep_z_and_axis,
    resample_data_or_seg_to_shape as _scipy_reference,
)


__all__ = [
    "resample_data_or_seg_to_shape_cucim",
    "resample_seg_channelargmax_to_shape_cucim",
    "resample_seg_volmatch_to_shape_cucim",
]


def _maybe_free_pool():
    """Release the CuPy memory pool unless NNUNET_CUCIM_KEEP_POOL is set.

    free_all_blocks() calls cudaFree, which synchronizes every stream and stalls
    the host thread. On a small GPU (e.g. the 6 GB dev box) freeing per call
    prevents fragmentation OOM, so it stays the default. On ample-VRAM boxes
    (A100/Blackwell) export NNUNET_CUCIM_KEEP_POOL=1 to keep the pool cached and
    skip the per-call driver stall; the caller is then responsible for bounding
    memory (e.g. one free per case)."""
    if os.environ.get("NNUNET_CUCIM_KEEP_POOL", "").lower() in ("1", "true", "yes"):
        return
    import cupy as cp
    cp.get_default_memory_pool().free_all_blocks()


# ---------------------------------------------------------------------------
# Pure CuPy primitives. CuPy in, CuPy out. Caller owns buffer lifecycle.
# Each primitive wraps its cuCIM kernel input in cp.ascontiguousarray to
# avoid a silent perf cliff on transposed or sliced views.
# ---------------------------------------------------------------------------
def _resize_image_gpu_in_out(vol_g, new_shape: Tuple[int, ...], order: int):
    """Image resize primitive. vol_g may be any dtype; output is float32."""
    import cupy as cp
    from cucim.skimage.transform import resize as _cucim_resize

    vol_g = cp.ascontiguousarray(vol_g, dtype=cp.float32)
    return _cucim_resize(
        vol_g, tuple(int(s) for s in new_shape),
        order=order, mode="edge", cval=0,
        anti_aliasing=False, preserve_range=True,
    )


def _resize_seg_perclass_gpu_in_out(vol_g, new_shape: Tuple[int, ...], order: int, out_dtype):
    """Per class binary resize + 0.5 threshold primitive.

    vol_g is a single channel seg (integer or float) on device. Iterates
    over cp.unique(vol_g) in ascending order so class 0 (background) is
    painted first and foreground classes win ties at the 0.5 threshold.
    Skipping cl == 0 is safe because seg_o is zero initialized.
    """
    import cupy as cp
    from cucim.skimage.transform import resize as _cucim_resize

    new_shape_t = tuple(int(s) for s in new_shape)
    seg_o = cp.zeros(new_shape_t, dtype=out_dtype)
    for cl in cp.asnumpy(cp.unique(vol_g)).tolist():
        if cl == 0:
            continue
        binary_g = cp.ascontiguousarray((vol_g == cl).astype(cp.float32))
        # clip=False on purpose. The m >= 0.5 threshold below is invariant to
        # clamping the resized values into [0, 1], so skipping clip avoids the
        # two host syncs cucim's _clip_warp_output does via input.min/max .item().
        m = _cucim_resize(
            binary_g, new_shape_t,
            order=order, mode="edge", cval=0,
            anti_aliasing=False, preserve_range=True, clip=False,
        )
        seg_o[m >= 0.5] = out_dtype.type(cl)
    return seg_o


def _isovalue_for_count_gpu(arr_g, target_n: int):
    """Device-resident isovalue for count matched thresholding.

    Returns the threshold t (a 0-d cupy array kept on device) such that
    (arr_g >= t) keeps exactly target_n voxels when t is unique. cp.partition
    does an O(n) selection on the GPU, so neither the sort nor the resulting
    membership threshold round trips to host. The only host sync is target_n,
    which the caller already has as a Python int.
    """
    import cupy as cp

    target_n = int(round(target_n))
    flat = arr_g.ravel()
    if target_n <= 0:
        return flat.max() + 1            # nothing crosses; keep zero voxels
    if target_n >= flat.size:
        return flat.min() - 1            # everything crosses; keep all voxels
    kth = flat.size - target_n
    return cp.partition(flat, kth)[kth]


def _resize_seg_volmatch_gpu_in_out(vol_g, new_shape: Tuple[int, ...], order: int,
                                    out_dtype, vol_scale: float, size_order: bool = False):
    """Per class volume preserving seg resize primitive.

    Like _resize_seg_perclass_gpu_in_out but replaces the fixed 0.5 threshold
    with a per class count matched threshold. Each foreground class is resized
    as a float membership (clip=False, same reasoning as the per class path) and
    thresholded at the isovalue that keeps round(native_class_count * vol_scale)
    voxels, so the resampled class volume equals the native class volume.
    vol_scale is prod(current_spacing) / prod(new_spacing), the voxel volume
    ratio that converts a native voxel count into the count occupying the same
    physical volume on the new grid.

    Ascending class iteration so background is painted first and a higher id
    class overwrites on overlap, matching _resize_seg_perclass_gpu_in_out. For
    spatially disjoint labels (the adrenal L/R glands) there is no overlap, so
    each class keeps its exact count; overlapping labels would have the later
    class shave voxels off the earlier one, the same overwrite policy as the
    per class path.
    """
    import cupy as cp
    from cucim.skimage.transform import resize as _cucim_resize

    new_shape_t = tuple(int(s) for s in new_shape)
    seg_o = cp.zeros(new_shape_t, dtype=out_dtype)
    # nnU-Net's GPU preprocessor carries the seg through as a float buffer, but
    # cp.bincount + the integer class ids below need an int label map. Seg values
    # are exact integer labels, so rint+cast is lossless. The cp.unique seg paths
    # in this file tolerate float; bincount does not -- which is why this only
    # surfaced once volmatch became the production seg resampler (the resampler-
    # bench CHECK_EQUIV fed int segs).
    if vol_g.dtype.kind not in "iu":
        vol_g = cp.rint(vol_g).astype(cp.int32)
    # All native class counts in ONE bincount pass: replaces cp.unique (a sort)
    # plus a per-class cls_mask.sum() that each forced a host sync. counts lives on
    # host (numpy) so native_count below is a plain int -> no per-class GPU sync.
    # Ascending class id (default) is byte-identical to the cp.unique iteration
    # (higher id overwrites on overlap). size_order=True instead paints the
    # largest-count classes FIRST so the smallest (adrenals, vessels) are painted
    # LAST and win contested boundary voxels -> near-zero volume drift on small
    # glands (the resampler-bench volmatch_sizeorder winner; rbench_methods.py).
    counts = cp.asnumpy(cp.bincount(vol_g.ravel()))
    class_ids = [c for c in range(1, len(counts)) if counts[c] > 0]
    if size_order:
        class_ids.sort(key=lambda c: counts[c], reverse=True)
    for cl in class_ids:
        native_count = int(counts[cl])
        binary_g = cp.ascontiguousarray((vol_g == cl).astype(cp.float32))
        m = _cucim_resize(
            binary_g, new_shape_t,
            order=order, mode="edge", cval=0,
            anti_aliasing=False, preserve_range=True, clip=False,
        )
        t = _isovalue_for_count_gpu(m, native_count * vol_scale)
        seg_o[m >= t] = out_dtype.type(cl)
    return seg_o


# ---------------------------------------------------------------------------
# Separate z helpers (matches upstream do_separate_z path)
# ---------------------------------------------------------------------------
def _build_z_coords(in_z: int, out_z: int, out_shape: Tuple[int, ...], axis: int):
    """Inverse coordinate map for 1D resample along `axis`, using nnU-Net's
    exact formula scale * (idx + 0.5) - 0.5."""
    import cupy as cp

    scales = [1.0] * 3
    scales[axis] = in_z / out_z
    grids = cp.meshgrid(*[cp.arange(n, dtype=cp.float32) for n in out_shape], indexing="ij")
    return cp.stack([scales[k] * (grids[k] + 0.5) - 0.5 for k in range(3)])


def _aniso_resample_one_gpu_in_out(vol_f32, new_shape_t: Tuple[int, ...], axis: int,
                                   order: int, order_z: int, coords=None, clip=None):
    """Anisotropic resample of one float32 volume.

    Per slice 2D in plane resize along `axis` into the intermediate shape,
    then a 1D map_coordinates along `axis`. Input and output are float32 cupy
    arrays. This is the single place the separate z kernel lives. The image
    primitive, the per class seg primitive, and the streaming channel argmax
    loop all call it.

    coords is the inverse z coordinate map from _build_z_coords. Callers that
    loop over classes pass a precomputed coords so the meshgrid builds once
    per case, not once per class. clip is forwarded to cucim resize. The image
    and channel argmax callers pass None so resize uses its order based default.
    The per class binary seg caller passes clip=False because its downstream
    >= 0.5 threshold is invariant to clamping into [0, 1], so skipping clip
    drops the host syncs cucim's _clip_warp_output does via .item().
    """
    import cupy as cp
    from cucim.skimage.transform import resize as _cucim_resize
    from cupyx.scipy.ndimage import map_coordinates as _cucim_map_coordinates

    in_z = int(vol_f32.shape[axis])
    inter_shape = tuple(int(s) if k != axis else in_z for k, s in enumerate(new_shape_t))
    new_shape_2d = tuple(s for k, s in enumerate(new_shape_t) if k != axis)
    if coords is None:
        coords = _build_z_coords(in_z, new_shape_t[axis], new_shape_t, axis)
    inter = cp.empty(inter_shape, dtype=cp.float32)
    for s in range(in_z):
        sl = (slice(None),) * axis + (s,)
        inter[sl] = _cucim_resize(
            cp.ascontiguousarray(vol_f32[sl]), new_shape_2d,
            order=order, mode="edge", cval=0,
            anti_aliasing=False, preserve_range=True, clip=clip,
        )
    return _cucim_map_coordinates(inter, coords, order=order_z, mode="nearest")


def _aniso_image_gpu_in_out(vol_g, new_shape: Tuple[int, ...], axis: int,
                            order: int, order_z: int):
    """Anisotropic image resample primitive (2D in-plane resize + 1D map_coordinates)."""
    import cupy as cp

    vol_g = cp.ascontiguousarray(vol_g, dtype=cp.float32)
    return _aniso_resample_one_gpu_in_out(
        vol_g, tuple(int(s) for s in new_shape), axis, order, order_z,
    )


def _aniso_seg_gpu_in_out(vol_g, new_shape: Tuple[int, ...], axis: int,
                          order: int, order_z: int, out_dtype):
    """Anisotropic seg resample primitive (per class 2D resize + 1D nearest in z)."""
    import cupy as cp

    new_shape_t = tuple(int(s) for s in new_shape)
    # Build the z coordinate map once, reused across every class.
    coords = _build_z_coords(vol_g.shape[axis], new_shape_t[axis], new_shape_t, axis)
    seg_o = cp.zeros(new_shape_t, dtype=out_dtype)
    for cl in cp.asnumpy(cp.unique(vol_g)).tolist():
        if cl == 0:
            continue   # background is no op since seg_o is zero initialized
        binary_g = (vol_g == cl).astype(cp.float32)
        # clip=False. See _aniso_resample_one_gpu_in_out. The bin_z >= 0.5
        # threshold is invariant to clamping, so skipping clip drops host syncs.
        bin_z = _aniso_resample_one_gpu_in_out(
            binary_g, new_shape_t, axis, order, order_z, coords=coords, clip=False,
        )
        seg_o[bin_z >= 0.5] = out_dtype.type(cl)
    return seg_o


def _channelargmax_streaming_gpu(
    vol_np: np.ndarray,
    new_shape: Tuple[int, ...],
    current_spacing,
    new_spacing,
    order: int,
    order_z: int,
    force_separate_z,
    separate_z_anisotropy_threshold: float,
    out_dtype: np.dtype,
    return_gpu: bool = False,
) -> np.ndarray:
    """Streaming GPU channel argmax.

    Replaces the per class loop in resample_seg_channelargmax_to_shape_cucim
    that round tripped each class through resample_data_or_seg_to_shape_cucim
    (CPU to GPU upload to GPU resize to GPU to CPU download to CPU argmax
    update to pool free). 21 to 26 of those per case dominated wall time.

    This helper uploads vol once and keeps it on GPU through the whole class
    loop, allocates best_logit and best_class on GPU, resizes each class's
    binary mask via the in process cucim primitives directly (no public
    wrapper re entry), updates best_* in place on GPU with strict `>` tie
    semantics so a class only wins a voxel when its resampled logit strictly
    exceeds the current best, copies best_class back to CPU exactly once, and
    leaves the single free_all_blocks() call to the caller.

    Algorithm equivalent to the old per class loop. Sorted ascending class
    iteration. Background included (cl=0 NOT skipped, see the public
    function docstring). Strict `>` tie break.

    return_gpu. When True, best_class is returned as a cupy.ndarray on the
    device and the caller takes ownership (no asnumpy, no pool flush). Used
    by _resample_seg_channelargmax_gpu_out so AdrenalGPUPreprocessor can
    keep the resampled seg on device through the foreground sampling step.
    Default False reproduces the original numpy return contract used by the
    public resample_seg_channelargmax_to_shape_cucim entry point.
    """
    import cupy as cp

    do_separate_z, axis = determine_do_sep_z_and_axis(
        force_separate_z, current_spacing, new_spacing, separate_z_anisotropy_threshold,
    )

    new_shape_t = tuple(int(s) for s in new_shape)
    vol_g = cp.asarray(vol_np)
    # nnU-Net's GPU preprocessor carries the seg through as a float buffer, but
    # cp.bincount needs an int array. Seg values are exact integer labels, so
    # rint+cast is lossless.
    if vol_g.dtype.kind not in "iu":
        vol_g = cp.rint(vol_g).astype(cp.int32)
    best_logit = cp.full(new_shape_t, -cp.inf, dtype=cp.float32)
    best_class = cp.zeros(new_shape_t, dtype=out_dtype)

    # Class list in ONE bincount pass (O(N)) instead of cp.unique (a sort).
    # Background (0) is included, ascending -- same set and order as cp.unique.
    classes = cp.asnumpy(cp.nonzero(cp.bincount(vol_g.ravel()))[0]).tolist()

    # Build the z coordinate map once per case. The per class resize and 1D z
    # resample live in _aniso_resample_one_gpu_in_out, the same primitive the
    # image and seg paths use. Non separate z classes go through the shared
    # isotropic image primitive.
    coords = (_build_z_coords(vol_g.shape[axis], new_shape_t[axis], new_shape_t, axis)
              if do_separate_z else None)

    for cl in classes:
        binary_g = (vol_g == cl).astype(cp.float32)
        if do_separate_z:
            chan_g = _aniso_resample_one_gpu_in_out(
                binary_g, new_shape_t, axis, order, order_z, coords=coords,
            )
        else:
            chan_g = _resize_image_gpu_in_out(binary_g, new_shape_t, order)
        # in place GPU argmax update. Strict > so lower id classes (background
        # first) hold a voxel on ties. cp.maximum is an in-place elementwise
        # kernel: it replaces best_logit[mask]=chan_g[mask], whose boolean GATHER
        # forced a host sync (to size the compacted array) every class. The scalar
        # scatter best_class[mask]=cl needs no sync (no gather). Byte-identical.
        mask = chan_g > best_logit
        cp.maximum(best_logit, chan_g, out=best_logit)
        best_class[mask] = out_dtype.type(cl)
        del binary_g, chan_g, mask

    if return_gpu:
        # Caller takes ownership of best_class. Do NOT asnumpy, do NOT free.
        del vol_g, best_logit
        if do_separate_z:
            del coords
        return best_class

    out = cp.asnumpy(best_class)
    del vol_g, best_logit, best_class
    if do_separate_z:
        del coords
    return out


def resample_logits_argmax_streaming_gpu(
    logits,
    new_shape: Tuple[int, ...],
    current_spacing,
    new_spacing,
    order: int = 1,
    order_z: int = 0,
    force_separate_z=None,
    separate_z_anisotropy_threshold: float = 3.0,
    out_dtype: np.dtype = np.dtype(np.uint8),
) -> np.ndarray:
    """Faithful resample-then-argmax of network LOGITS, streamed one channel at a time.

    nnU-Net's convert path resamples the FULL (C, *native) probability tensor (linear,
    order=1) and THEN argmaxes -- correct, but for a large multi-class CT that tensor is
    tens of GB and OOMs the GPU (or forces a slow all-channel CPU resample). This is the
    SAME operation, channel-streamed: keep best_logit (-inf) and best_class on device at
    the native shape, resample each logit channel prep->native with the SAME in-process
    cuCIM primitive the image/seg paths use, and argmax-accumulate in place (strict `>`
    so the lowest-index class holds a voxel on ties, exactly like argmax). VRAM is O(1
    channel + 2 native scalar grids), never O(C*native). Numerically equivalent to
    resample(logits, order=1)->argmax (float boundary noise only).

    This is the FAITHFUL replacement for the argmax-first shortcut (argmax at the 1 mm
    grid, then nearest-resample the integer labels), which is biased for SMALL structures
    -- and the adrenals are exactly that. Sibling of _channelargmax_streaming_gpu, which
    streams one-hot SEG masks for the seg resampler; here we stream the raw float logit
    channels directly (no binary masking), at order=1 like resampling_fn_probabilities.
    """
    import cupy as cp

    do_separate_z, axis = determine_do_sep_z_and_axis(
        force_separate_z, current_spacing, new_spacing, separate_z_anisotropy_threshold,
    )
    new_shape_t = tuple(int(s) for s in new_shape)
    n_ch = int(logits.shape[0])
    best_logit = cp.full(new_shape_t, -cp.inf, dtype=cp.float32)
    best_class = cp.zeros(new_shape_t, dtype=out_dtype)
    coords = (_build_z_coords(int(logits.shape[1 + axis]), new_shape_t[axis], new_shape_t, axis)
              if do_separate_z else None)

    for c in range(n_ch):
        # one channel -> device (O(1) upload even when `logits` lives on the host because
        # nnU-Net CPU-accumulated a volume too big to keep all channels on the GPU).
        chan_in = cp.ascontiguousarray(cp.asarray(logits[c], dtype=cp.float32))
        if do_separate_z:
            chan_g = _aniso_resample_one_gpu_in_out(chan_in, new_shape_t, axis, order, order_z, coords=coords)
        else:
            chan_g = _resize_image_gpu_in_out(chan_in, new_shape_t, order)
        # in-place argmax update, strict `>` (lower-index class wins ties, == np.argmax).
        mask = chan_g > best_logit
        cp.maximum(best_logit, chan_g, out=best_logit)
        best_class[mask] = out_dtype.type(c)
        del chan_in, chan_g, mask

    out = cp.asnumpy(best_class)
    del best_logit, best_class
    if do_separate_z:
        del coords
    _maybe_free_pool()
    return out


def _channelargmax_streaming_cpu(
    vol_np: np.ndarray,
    new_shape: Tuple[int, ...],
    current_spacing,
    new_spacing,
    order: int,
    order_z: int,
    force_separate_z,
    separate_z_anisotropy_threshold: float,
    out_dtype: np.dtype,
) -> np.ndarray:
    """CPU numpy mirror of _channelargmax_streaming_gpu.

    The channel argmax seg resampler routes here when NNUNET_CUCIM_BACKEND=cpu
    so it has a real CPU fallback instead of forcing a cupy import on a GPU
    free box. Same algorithm as the GPU path. Sorted ascending classes,
    background included in the argmax, per class resize at `order`, strict `>`
    tie break, and the separate z branch of a 2D in plane resize per slice
    plus a 1D map_coordinates at `order_z`. Uses skimage resize and scipy
    map_coordinates, the same kernels cucim mirrors on the GPU, so the output
    matches the GPU path to within float32 boundary noise.
    """
    from skimage.transform import resize as _sk_resize
    from scipy.ndimage import map_coordinates as _sp_map_coordinates

    do_separate_z, axis = determine_do_sep_z_and_axis(
        force_separate_z, current_spacing, new_spacing, separate_z_anisotropy_threshold,
    )
    new_shape_t = tuple(int(s) for s in new_shape)
    vol = np.asarray(vol_np)
    best_logit = np.full(new_shape_t, -np.inf, dtype=np.float32)
    best_class = np.zeros(new_shape_t, dtype=out_dtype)
    classes = sorted(int(v) for v in np.unique(vol))

    if do_separate_z:
        in_z = int(vol.shape[axis])
        scales = [1.0, 1.0, 1.0]
        scales[axis] = in_z / new_shape_t[axis]
        grids = np.meshgrid(*[np.arange(n, dtype=np.float32) for n in new_shape_t], indexing="ij")
        coords = np.stack([scales[k] * (grids[k] + 0.5) - 0.5 for k in range(3)])
        inter_shape = tuple(int(s) if k != axis else in_z for k, s in enumerate(new_shape_t))
        new_shape_2d = tuple(s for k, s in enumerate(new_shape_t) if k != axis)

    for cl in classes:
        binary = (vol == cl).astype(np.float32)
        if do_separate_z:
            inter = np.empty(inter_shape, dtype=np.float32)
            for s in range(in_z):
                sl = (slice(None),) * axis + (s,)
                inter[sl] = _sk_resize(
                    binary[sl], new_shape_2d, order=order, mode="edge", cval=0,
                    anti_aliasing=False, preserve_range=True,
                )
            chan = _sp_map_coordinates(
                inter, coords, order=order_z, mode="nearest",
            ).astype(np.float32)
        else:
            chan = _sk_resize(
                binary, new_shape_t, order=order, mode="edge", cval=0,
                anti_aliasing=False, preserve_range=True,
            ).astype(np.float32)
        # Strict > so lower id classes (background first) hold a voxel on ties.
        m = chan > best_logit
        best_logit[m] = chan[m]
        best_class[m] = out_dtype.type(cl)
    return best_class


def _resize_seg_volmatch_cpu(vol: np.ndarray, new_shape: Tuple[int, ...], order: int,
                             out_dtype, vol_scale: float, size_order: bool = False) -> np.ndarray:
    """CPU numpy mirror of _resize_seg_volmatch_gpu_in_out.

    Routes here when NNUNET_CUCIM_BACKEND=cpu so the volume match seg resampler
    has a real CPU fallback. np.partition selects the same kth value cp.partition
    does on the GPU, and skimage resize matches cucim resize within float noise,
    so the count matched output agrees with the GPU path up to boundary ties.
    """
    from skimage.transform import resize as _sk_resize

    new_shape_t = tuple(int(s) for s in new_shape)
    seg_o = np.zeros(new_shape_t, dtype=out_dtype)
    # Mirror the GPU path: ascending class id by default; size_order paints the
    # largest-count classes first so the smallest structures win contested ties.
    class_ids = [int(c) for c in np.unique(vol).tolist() if c != 0]
    if size_order:
        cnt = {c: int((vol == c).sum()) for c in class_ids}
        class_ids.sort(key=lambda c: cnt[c], reverse=True)
    for cl in class_ids:
        cls_mask = vol == cl
        native_count = int(cls_mask.sum())
        binary = cls_mask.astype(np.float32)
        m = _sk_resize(
            binary, new_shape_t, order=order, mode="edge", cval=0,
            anti_aliasing=False, preserve_range=True,
        )
        target_n = int(round(native_count * vol_scale))
        flat = m.ravel()
        if target_n <= 0:
            t = flat.max() + 1
        elif target_n >= flat.size:
            t = flat.min() - 1
        else:
            kth = flat.size - target_n
            t = np.partition(flat, kth)[kth]
        seg_o[m >= t] = out_dtype.type(cl)
    return seg_o


# ---------------------------------------------------------------------------
# Device-resident orchestrator. CuPy in, CuPy out. Used directly by
# AdrenalGPUPreprocessor._dispatch_data_resample_gpu so the full pipeline
# stays on device between crop, normalize, and FG sampling.
# ---------------------------------------------------------------------------
def _resample_data_or_seg_to_shape_cucim_gpu_in_out(
    data_g,
    new_shape: Union[Tuple[int, ...], List[int], np.ndarray],
    current_spacing,
    new_spacing,
    is_seg: bool = False,
    order: int = 3,
    order_z: int = 0,
    force_separate_z: Union[bool, None] = False,
    separate_z_anisotropy_threshold: float = 3.0,
    **_ignored,
):
    """Device-resident twin of resample_data_or_seg_to_shape_cucim.

    Accepts a CuPy ndarray of shape (C, X, Y, Z) and returns a CuPy ndarray
    of shape (C, X', Y', Z'). No host upload, no host download, no memory
    pool flush. Caller owns the returned buffer.

    Anisotropy decision and per-channel orchestration are identical to the
    public NumPy wrapper. The two surfaces share every primitive so any
    drift between them is by definition impossible.
    """
    import cupy as cp

    assert data_g is not None and data_g.ndim == 4, "data_g must be (c, x, y, z)"
    new_shape_t = tuple(int(s) for s in new_shape)

    do_separate_z, axis = determine_do_sep_z_and_axis(
        force_separate_z, current_spacing, new_spacing, separate_z_anisotropy_threshold,
    )

    in_shape = tuple(int(s) for s in data_g.shape[1:])
    out_dtype = data_g.dtype if is_seg else cp.float32

    if in_shape == new_shape_t:
        return data_g.astype(out_dtype, copy=False)

    out_g = cp.empty((data_g.shape[0], *new_shape_t), dtype=out_dtype)
    for c in range(data_g.shape[0]):
        vol = data_g[c]
        if is_seg:
            if do_separate_z:
                out_g[c] = _aniso_seg_gpu_in_out(vol, new_shape_t, axis, order, order_z, out_dtype)
            else:
                out_g[c] = _resize_seg_perclass_gpu_in_out(vol, new_shape_t, order, out_dtype)
        else:
            if do_separate_z:
                out_g[c] = _aniso_image_gpu_in_out(vol, new_shape_t, axis, order, order_z)
            else:
                out_g[c] = _resize_image_gpu_in_out(vol, new_shape_t, order)
    return out_g


# ---------------------------------------------------------------------------
# Public entry point with the same signature as nnU-Net's reference
# ---------------------------------------------------------------------------
def resample_data_or_seg_to_shape_cucim(
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
    """GPU accelerated drop in for `resample_data_or_seg_to_shape`.
    See module docstring for semantics and fallback behavior.

    Implementation, single primitive + two surfaces.

    This public wrapper uploads `data` to device, delegates the actual work
    to `_resample_data_or_seg_to_shape_cucim_gpu_in_out` (which orchestrates
    the shared per channel primitives), downloads the result, and frees the
    cupy memory pool. The same orchestrator is called directly by
    AdrenalGPUPreprocessor when the device-resident path is active, which
    saves the upload and the download.
    """
    # Optional CPU fallback for multi process inference or no GPU machines.
    # Resolved BEFORE importing cupy so the fallback path runs on a CPU only box.
    if os.environ.get("NNUNET_CUCIM_BACKEND", "").lower() == "cpu":
        return _scipy_reference(
            data, new_shape, current_spacing, new_spacing,
            is_seg=is_seg, order=order, order_z=order_z,
            force_separate_z=force_separate_z,
            separate_z_anisotropy_threshold=separate_z_anisotropy_threshold,
        )

    import cupy as cp

    # torch to numpy (mirrors upstream)
    try:
        import torch  # type: ignore
        if isinstance(data, torch.Tensor):
            data = data.numpy()
    except Exception:
        pass

    assert data is not None and data.ndim == 4, "data must be (c, x, y, z)"
    new_shape_t = tuple(int(s) for s in new_shape)
    in_shape = tuple(int(s) for s in data.shape[1:])

    # Identity short circuit. Matches upstream: returns input unchanged when
    # shapes already match, after dtype coercion to the expected output type.
    out_dtype = data.dtype if is_seg else np.float32
    if in_shape == new_shape_t:
        return data.astype(out_dtype, copy=False)

    data_g = cp.asarray(data)
    out_g = _resample_data_or_seg_to_shape_cucim_gpu_in_out(
        data_g, new_shape_t, current_spacing, new_spacing,
        is_seg=is_seg, order=order, order_z=order_z,
        force_separate_z=force_separate_z,
        separate_z_anisotropy_threshold=separate_z_anisotropy_threshold,
    )
    out = cp.asnumpy(out_g)
    del data_g, out_g
    _maybe_free_pool()
    return out


# ---------------------------------------------------------------------------
# Channel argmax seg resampler. The path that measured 0.998 Dice in the v3
# round trip nn_fwdLogit_backLogit test versus around 0.96 for per class
# binary plus 0.5 threshold. Plan callable with the same signature as
# resample_data_or_seg_to_shape.
# ---------------------------------------------------------------------------
def resample_seg_channelargmax_to_shape_cucim(
    data,
    new_shape,
    current_spacing,
    new_spacing,
    order: int = 3,
    order_z: int = 0,
    force_separate_z=False,
    separate_z_anisotropy_threshold: float = 3.0,
    **_ignored,   # absorb e.g. is_seg if nnU-Net forwards it from kwargs
):
    """Channel argmax label resampler.

    Internally one hot encodes the integer label map, runs per channel
    bspline N (order=3 by default) via the image path of
    `resample_data_or_seg_to_shape_cucim` (`is_seg=False`), then takes a
    streaming argmax across classes. Matches the production faithful nnU-Net
    export semantics on the back step and matches the 0.998 Dice round trip
    measured in tests/test_roundtrip_native.py.

    Background (cl == 0) is INCLUDED in the argmax. Its bspline resampled
    background logit must compete with foreground class logits at boundaries.
    The Test 1 skip cl=0 optimization, which IS bit identical for the
    threshold and overwrite pattern, is NOT safe here. Skipping background
    would let foreground classes win in voxels that should be background.

    Plan callable surface (nnU-Net plans file).
        "resampling_fn_seg": "resample_seg_channelargmax_to_shape_cucim",
        "resampling_fn_seg_kwargs": {"order": 3, "order_z": 0,
                                      "force_separate_z": null}

    CPU fallback. When NNUNET_CUCIM_BACKEND=cpu this routes to
    _channelargmax_streaming_cpu, a numpy mirror of the GPU path, so the same
    channel argmax algorithm runs on a GPU free box without importing cupy.
    AdrenalGPUPreprocessor's _cpu_resample_fallback calls this function for the
    seg path when the env var is set.
    """
    # torch to numpy (mirrors upstream). No cupy needed for this or the CPU path.
    try:
        import torch  # type: ignore
        if isinstance(data, torch.Tensor):
            data = data.numpy()
    except Exception:
        pass

    assert data is not None and data.ndim == 4, "data must be (c, x, y, z)"
    assert data.shape[0] == 1, "seg input must be single channel"
    new_shape = tuple(int(s) for s in new_shape)
    in_shape = tuple(int(s) for s in data.shape[1:])
    out_dtype = np.dtype(data.dtype)

    if in_shape == new_shape:
        return data.astype(out_dtype, copy=False)

    # CPU fallback resolved BEFORE importing cupy so it runs on a GPU free box.
    if os.environ.get("NNUNET_CUCIM_BACKEND", "").lower() == "cpu":
        best_class = _channelargmax_streaming_cpu(
            data[0], new_shape, current_spacing, new_spacing,
            order, order_z, force_separate_z, separate_z_anisotropy_threshold,
            out_dtype,
        )
        return best_class[None]

    import cupy as cp
    # Streaming GPU argmax. One upload, per class GPU resize plus GPU argmax
    # update, one download.
    best_class = _channelargmax_streaming_gpu(
        data[0], new_shape,
        current_spacing, new_spacing,
        order, order_z,
        force_separate_z, separate_z_anisotropy_threshold,
        out_dtype,
    )

    _maybe_free_pool()
    return best_class[None]   # (1, *new_shape)


# ---------------------------------------------------------------------------
# Volume preserving (count matched) seg resampler. The training cache leg
# optimization from the volume match study: per class count matched threshold
# removes the forward volume drift the fixed 0.5 threshold leaves on small
# glands, holding Dice. Plan callable with the resample_data_or_seg_to_shape
# signature.
# ---------------------------------------------------------------------------
def resample_seg_volmatch_to_shape_cucim(
    data,
    new_shape,
    current_spacing,
    new_spacing,
    order: int = 3,
    order_z: int = 0,
    force_separate_z=False,
    separate_z_anisotropy_threshold: float = 3.0,
    size_order: bool = False,
    **_ignored,   # absorb e.g. is_seg if nnU-Net forwards it from kwargs
):
    """Volume preserving per class seg resampler (GPU, count matched threshold).

    Resamples each foreground class as a float membership and thresholds at the
    isovalue that keeps the native class voxel count scaled by the voxel volume
    ratio prod(current_spacing) / prod(new_spacing), so each class's physical
    volume is preserved across the resample. This is the training cache leg
    optimization from the volume match study: it removes the forward volume drift
    the fixed 0.5 threshold per class resampler leaves on small adrenal glands,
    while holding Dice. order defaults to 3 (cubic membership). An order sweep on
    239 adrenal glands showed cubic gives a tighter count match than linear (its
    denser value spectrum removes the tie overshoot that order 1 leaves on the
    threshold) and better boundary fidelity, with the largest gain on small
    glands and no fragmentation cost (cubic fragmented fewer glands than linear).
    order 0 is unusable here: a binary membership field cannot be count matched.

    Plan callable surface (nnU-Net plans file):
        "resampling_fn_seg": "resample_seg_volmatch_to_shape_cucim",
        "resampling_fn_seg_kwargs": {"order": 3, "force_separate_z": false}

    Only the isotropic (force_separate_z=False) path is supported, which is the
    pinned adrenal configuration. A separate z request raises NotImplementedError
    rather than silently count matching along an axis the volume preservation
    derivation does not cover.

    CPU fallback. NNUNET_CUCIM_BACKEND=cpu routes to _resize_seg_volmatch_cpu, a
    numpy mirror, so the same count matched algorithm runs on a GPU free box.
    """
    # torch to numpy (mirrors upstream). No cupy needed for this or the CPU path.
    try:
        import torch  # type: ignore
        if isinstance(data, torch.Tensor):
            data = data.numpy()
    except Exception:
        pass

    assert data is not None and data.ndim == 4, "data must be (c, x, y, z)"
    new_shape_t = tuple(int(s) for s in new_shape)
    in_shape = tuple(int(s) for s in data.shape[1:])
    out_dtype = np.dtype(data.dtype)

    if in_shape == new_shape_t:
        return data.astype(out_dtype, copy=False)

    # CPU fallback resolved BEFORE importing cupy so it runs on a GPU free box.
    if os.environ.get("NNUNET_CUCIM_BACKEND", "").lower() == "cpu":
        do_separate_z, _axis = determine_do_sep_z_and_axis(
            force_separate_z, current_spacing, new_spacing, separate_z_anisotropy_threshold,
        )
        if do_separate_z:
            raise NotImplementedError(
                "resample_seg_volmatch_to_shape_cucim supports only the isotropic "
                "(force_separate_z=False) path; got a separate z resample."
            )
        vol_scale = float(np.prod(current_spacing)) / float(np.prod(new_spacing))
        return np.stack([
            _resize_seg_volmatch_cpu(data[c], new_shape_t, order, out_dtype, vol_scale, size_order)
            for c in range(data.shape[0])
        ])

    import cupy as cp
    out_g = _resample_seg_volmatch_gpu_out(
        data, new_shape_t, current_spacing, new_spacing,
        order, order_z, force_separate_z, separate_z_anisotropy_threshold,
        size_order,
    )
    out = cp.asnumpy(out_g)
    del out_g
    _maybe_free_pool()
    return out


def _resample_seg_volmatch_gpu_out(
    data,
    new_shape: Union[Tuple[int, ...], List[int], np.ndarray],
    current_spacing: Union[Tuple[float, ...], List[float], np.ndarray],
    new_spacing: Union[Tuple[float, ...], List[float], np.ndarray],
    order: int = 3,
    order_z: int = 0,
    force_separate_z: Union[bool, None] = False,
    separate_z_anisotropy_threshold: float = 3.0,
    size_order: bool = False,
    **_ignored,
):
    """INTERNAL volume-preserving per class seg resampler that returns on the GPU.

    Identical algorithm to resample_seg_volmatch_to_shape_cucim.
    Difference: The result is NOT downloaded to host and the cupy memory
    pool is NOT flushed. Caller takes ownership of the returned cupy.ndarray.
    """
    import cupy as cp

    try:
        import torch  # type: ignore
        if isinstance(data, torch.Tensor):
            data = data.numpy()
    except Exception:
        pass

    assert data is not None and data.ndim == 4, "data must be (c, x, y, z)"
    new_shape_t = tuple(int(s) for s in new_shape)
    in_shape = tuple(int(s) for s in data.shape[1:])
    out_dtype = np.dtype(data.dtype)

    if in_shape == new_shape_t:
        return cp.asarray(data, dtype=out_dtype)

    do_separate_z, _axis = determine_do_sep_z_and_axis(
        force_separate_z, current_spacing, new_spacing, separate_z_anisotropy_threshold,
    )
    if do_separate_z:
        raise NotImplementedError(
            "resample_seg_volmatch_to_shape_cucim supports only the isotropic "
            "(force_separate_z=False) path; got a separate z resample."
        )

    vol_scale = float(np.prod(current_spacing)) / float(np.prod(new_spacing))

    data_g = cp.asarray(data)
    out_g = cp.empty((data.shape[0], *new_shape_t), dtype=out_dtype)
    for c in range(data.shape[0]):
        out_g[c] = _resize_seg_volmatch_gpu_in_out(
            data_g[c], new_shape_t, order, out_dtype, vol_scale, size_order,
        )
    return out_g


# ---------------------------------------------------------------------------
# Internal device-returning twin of resample_seg_channelargmax_to_shape_cucim.
# Used by AdrenalGPUPreprocessor to keep the resampled seg on device through
# the foreground sampling step. Leading underscore is a SOCIAL convention
# (matches the other _helpers in this file). nnU-Net's
# recursive_find_python_class walks via hasattr and does not filter on
# underscore names, so a plans JSON that named this string would technically
# resolve. The underscore is the documented marker that this is direct-import
# only:
#   from nnunetv2.preprocessing.resampling.cucim_resampling import (
#       _resample_seg_channelargmax_gpu_out,
#   )
# ---------------------------------------------------------------------------
def _resample_seg_channelargmax_gpu_out(
    data,
    new_shape: Union[Tuple[int, ...], List[int], np.ndarray],
    current_spacing: Union[Tuple[float, ...], List[float], np.ndarray],
    new_spacing: Union[Tuple[float, ...], List[float], np.ndarray],
    order: int = 3,
    order_z: int = 0,
    force_separate_z: Union[bool, None] = False,
    separate_z_anisotropy_threshold: float = 3.0,
    **_ignored,
):
    """INTERNAL channel argmax label resampler that returns on the GPU.

    Identical algorithm to resample_seg_channelargmax_to_shape_cucim.
    Difference. The result is NOT downloaded to host and the cupy memory
    pool is NOT flushed. Caller takes ownership of the returned cupy.ndarray
    and must free it (typically via del plus cp.get_default_memory_pool().free_all_blocks()
    once it is done with the device buffer).

    Designed for use inside AdrenalGPUPreprocessor.run_case_npy, where the
    next operation on seg is a GPU foreground sampler. Saves one
    D2H copy plus one H2D copy per case versus going through the public
    function and then re uploading inside the sampler.

    Parameters mirror resample_seg_channelargmax_to_shape_cucim. data must
    be (1, X, Y, Z) and may be either a numpy.ndarray (legacy callers that
    still hand a host array) or a cupy.ndarray (the device-resident path).
    Returns shape (1, X', Y', Z') as a cupy.ndarray with dtype == data.dtype.
    Identity branch (in_shape == new_shape) returns cp.asarray(data,
    dtype=out_dtype) so the caller always receives a GPU array.

    The underscore prefix is convention. recursive_find_resampling_fn_by_name
    (in nnunetv2/preprocessing/resampling/utils.py line 8) walks the directory
    via hasattr, which does not filter on leading underscore. So a plans JSON
    that named this string would technically resolve. The underscore is the
    documented social marker that this is direct-import only.
    """
    import cupy as cp

    try:
        import torch  # type: ignore
        if isinstance(data, torch.Tensor):
            data = data.numpy()
    except Exception:
        pass

    assert data is not None and data.ndim == 4, "data must be (c, x, y, z)"
    assert data.shape[0] == 1, "seg input must be single channel"
    new_shape_t = tuple(int(s) for s in new_shape)
    in_shape = tuple(int(s) for s in data.shape[1:])
    # data.dtype works for both numpy and cupy arrays (both expose .dtype).
    out_dtype = np.dtype(data.dtype)

    if in_shape == new_shape_t:
        # Identity. Promote to GPU and return with the channel dim intact.
        return cp.asarray(data, dtype=out_dtype)

    # _channelargmax_streaming_gpu accepts either numpy or cupy via cp.asarray.
    # For the device-resident path data already lives on GPU; cp.asarray is a
    # no-op (returns the same buffer). For legacy callers passing numpy we
    # still pay one H2D, which is the historical contract.
    best_class_g = _channelargmax_streaming_gpu(
        data[0], new_shape_t,
        current_spacing, new_spacing,
        order, order_z,
        force_separate_z, separate_z_anisotropy_threshold,
        out_dtype,
        return_gpu=True,
    )
    return best_class_g[None]   # (1, *new_shape) on device
