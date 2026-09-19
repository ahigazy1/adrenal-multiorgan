# Vendored nnU-Net

`vendor/nnUNet` is an unmodified copy of https://github.com/MIC-DKFZ/nnUNet at commit
`ded2aa3a4c81a9caae37054224d8ebac2d19a061` (version 2.8.1, Apache-2.0; see `nnUNet/LICENSE`), plus two added files:

| Added file (in `nnUNet/nnunetv2/preprocessing/resampling/`) | SHA-256 | What it provides |
|---|---|---|
| `sitk_logits_resampling.py` | `442f377746bfcc618e8e97ed47e5bfa5f19a18a58087145a59c10b43943c9e54` | `resample_logits_to_shape_sitk`: resamples network output back to the original grid with SimpleITK. Reproduces nnU-Net's default result to float precision, multithreaded. |
| `sitk_resampling.py` | `d4cdf2d2177bc6aa631812fa56d3a8f4df47d367dc64d10586377df4d4c56090` | `resample_data_or_seg_to_shape_sitk`: resamples images or labels in physical space with SimpleITK. Does not reproduce nnU-Net's default; see PLAN.md, open item 1. |

Both were written by this repository's author for earlier work. No upstream file was changed. nnU-Net finds resamplers by function name, so they are
switched on only by naming them in a plans file (`resampling_fn_data`, `resampling_fn_probabilities`).

Install with `pip install -e vendor/nnUNet`.
