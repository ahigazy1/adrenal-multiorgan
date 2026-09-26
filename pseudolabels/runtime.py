"""CPU resampling and bounded GPU admission for the pinned TotalSegmentator API."""
import gc
import inspect
import os
from functools import wraps

import numpy as np

THREADS = 2


def resample_img(img, zoom=0.5, order=0, nr_cpus=THREADS):
    """Replace TS's array zoom, retaining its affine/orientation/crop handling.

    Match scipy.ndimage.zoom(grid_mode=False): first and last voxel centres
    coincide, including singleton axes. These are array-index coordinates,
    not NIfTI RAS or SimpleITK LPS world coordinates. TS handles world geometry.
    Only the linear CT and nearest-label paths used by this recipe are allowed.
    """
    import SimpleITK as sitk
    if img.ndim != 3 or order not in (0, 1):
        raise ValueError('Pseudo-label resampling requires 3D data and order 0 or 1')
    shape = np.asarray(img.shape)
    target = np.round(shape * np.broadcast_to(zoom, (3,))).astype(int)
    if np.any(target < 1):
        raise ValueError('Resampling would produce an empty axis')
    if np.array_equal(shape, target):
        return img.copy()
    spacing = np.divide(shape - 1, target - 1, out=np.ones(3), where=target > 1)
    # A singleton input axis must sample index zero while other axes interpolate.
    # Whole-pixel nearest extrapolation would incorrectly round the other axes.
    spacing[shape == 1] = 1
    source = sitk.GetImageFromArray(np.ascontiguousarray(img))
    filt = sitk.ResampleImageFilter()
    filt.SetNumberOfThreads(THREADS)
    filt.SetNumberOfWorkUnits(THREADS)
    filt.SetSize([int(v) for v in target[::-1]])
    filt.SetOutputSpacing([float(v) for v in spacing[::-1]])
    transform = sitk.AffineTransform(3)
    transform.SetMatrix(np.diag((shape[::-1] > 1).astype(float)).ravel().tolist())
    filt.SetTransform(transform)
    filt.SetInterpolator(sitk.sitkNearestNeighbor if order == 0 else sitk.sitkLinear)
    filt.SetUseNearestNeighborExtrapolator(True)
    return sitk.GetArrayFromImage(filt.Execute(source))


def gated_predict(predict, gate):
    """Serialize model loading/inference/export, releasing cached GPU allocations."""
    signature = inspect.signature(predict)

    @wraps(predict)
    def call(*args, **kwargs):
        import torch
        bound = signature.bind(*args, **kwargs)
        # TS passes resampling threads as a preprocessing PROCESS count.
        # Keep nested pools small; the outer case pool supplies concurrency.
        bound.arguments['num_threads_preprocessing'] = 1
        bound.arguments['num_threads_nifti_save'] = 1
        with gate:
            try:
                return predict(*bound.args, **bound.kwargs)
            finally:
                gc.collect()
                torch.cuda.empty_cache()
    return call


def setup(gate):
    # Set before importing nnU-Net: export defaults otherwise use 8 threads.
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                'SITK_THREADS', 'ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS', 'nnUNet_def_n_proc'):
        os.environ[key] = str(THREADS)
    import torch
    import SimpleITK as sitk
    from totalsegmentator import nnunet, resampling
    torch.set_num_threads(THREADS)
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(THREADS)
    resampling.resample_img = resample_img
    # Route the optional cuCIM path to the same CPU implementation as well.
    resampling.resample_img_cucim = resample_img
    nnunet.nnUNetv2_predict = gated_predict(nnunet.nnUNetv2_predict, gate)
