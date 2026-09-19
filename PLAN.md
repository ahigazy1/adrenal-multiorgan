# Specification

Last updated September 19, 2026. Status: all five steps and the Google Cloud scripts are written; steps 1-3 and the trainer are checked locally on CPU; nothing has been run on Colab or a GPU yet. Results used for publication come only from logged Colab and GPU runs. Local runs are for testing code.

## Setup

Data preparation (download, cleaning, preprocessing, publishing the cache to Hugging Face) runs on a Google Colab runtime. Training runs on one cloud VM with an NVIDIA RTX PRO 6000 (96 GB) and 48 vCPUs. Training is a single command that starts or continues by itself, because the VM may be interrupted at any time.

## Decided

| Item | Value |
|---|---|
| Sources | all of TotalSegmentator v2.0.1 CT (1228), AMOS22 CT (300 labelled), BTCV (30) and FLARE22 (50 labelled), with their public labels |
| Labels (9 foreground) | adrenal left/right, kidney left/right, liver, spleen, aorta, inferior vena cava, pancreas |
| Kidney cysts | TotalSegmentator's separate cyst masks are merged into the kidney (AMOS and BTCV have no cyst class) |
| Scans without any target organ | kept |
| Spacing | 1.0 mm isotropic |
| Architecture and initialization | AtlasNet (nnU-Net ResEncL, patch 256x128x192); encoder and decoder weights transferred, segmentation heads new |
| Batch size, epochs, fold | 4, 2000, fold 0 of one fixed split |
| Schedule | SGD (momentum 0.99, Nesterov, weight decay 3e-5); peak LR 1e-3; 50 epochs linear warmup; then polynomial decay, exponent 0.9 |
| Mirroring | none, in training or inference |
| Augmentation | reduced: no blur, noise or simulated low resolution; rotation +/-10 degrees and scaling 0.9-1.1, each with probability 0.1 |
| Loss | cross-entropy + focal Tversky (false negatives 0.6, false positives 0.4, exponent 0.75), deep supervision |
| Resampling | SimpleITK for both: `resample_data_or_seg_to_shape_sitk` prepares CT and labels for the training cache and CT at inference; `resample_logits_to_shape_sitk` brings the network output back to the original grid |
| CT intensity normalization | AtlasNet's, unchanged: clip to [-1000, 629] HU, subtract -182.95, divide by 406.48 (the values stored in `atlasnet_plans.json`). They are not recomputed from our data, because the transferred weights expect inputs on this scale |
| Image reader | nnU-Net's `NibabelIOWithReorient`, as in AtlasNet's plans |
| Test, validation, training | 20% / 16% / 64% of the kept scans of every source, stratified by slice thickness and adrenal volume (see below); validation is used for monitoring and for choosing the best checkpoint |
| Cache | private Hugging Face dataset `ahigazy1/adrenal-multiorgan-cache` |
| Checkpoints | every 100 epochs, plus the best (by nnU-Net's validation Dice moving average); uploaded to Hugging Face every 100 epochs |

### Adrenal label cleaning rule

Applied per gland, to every source, in training and test data. Connected components are 26-connected on the original label grid; volumes use the original voxel size.

1. Gland absent: the scan is kept.
2. Secondary components smaller than 0.3 mL are speckles and are removed.
3. A secondary component of 0.3 mL or more means the label is broken: the whole scan is excluded.
4. The remaining component must be larger than 1 mL, otherwise the whole scan is excluded.
5. A gland that touches any edge of the scan volume is cut off, so its volume is not the gland's volume: the whole scan is excluded.
6. Left/right check, in world coordinates: the left adrenal and left kidney must lie to the patient's left of the right ones, and the spleen to the left of the liver (only organs that are present are compared). A scan that fails is excluded. This catches swapped labels, a wrong orientation in the file header and mirrored anatomy.
7. Every removal and exclusion is recorded per case.

Excluding the whole scan (rather than masking one gland) keeps the rule one sentence long. In a local test on 150 TotalSegmentator scans, rules 3-5 together excluded 17% (rules 3-4 alone about 8%); none failed the left/right check.

### Learning-rate schedule: why this one

It is nnU-Net's own fine-tuning recipe, unchanged (`PretrainedTrainer` and `nnUNetTrainer_warmup` in the vendored code: SGD, peak 1e-3 when pretrained weights are loaded versus 1e-2 from scratch, 50 warmup epochs, offset polynomial decay). Not chosen: a higher peak (in preliminary 100-epoch runs 5e-3 was ahead of 1e-3 only within single-run noise, and the reason for it, too few updates, does not apply to 2000 epochs); scaling the rate with batch size (nnU-Net never does); cosine schedules, restarts, layer-wise rates or freezing (not standard in nnU-Net, no evidence they are needed).

### Held-out test set and training split

No dataset's published split is used. All kept scans are divided by one rule, so every source contributes the same share and the parts have the same mix:

- Scans are grouped by source, slice thickness (<=2 mm, 2-4 mm, >4 mm; the coarsest voxel axis) and total adrenal volume (none, or the lower, middle or upper third of scans that have adrenals). Groups with fewer than 10 scans fall back to source and volume, then to source alone.
- scikit-learn's `StratifiedKFold` (5 folds, shuffled, seed 12345) divides every group evenly. One fifth of all kept scans is the held-out test set.
- The remaining four fifths are divided the same way into five folds for nnU-Net; fold 0 is trained, so its validation part is one fifth of the remainder (16% of all scans) and its training part is 64%.
- `build_dataset.py` logs and saves the size, adrenal-volume median, slice-thickness mix and per-group counts of each part.

TotalSegmentator is distributed at 1.5 mm isotropic, so slice-thickness variety comes from AMOS, BTCV and FLARE22 only. Because published test splits are not used, results are not directly comparable with numbers other papers report on those splits. Limitation that stays: AtlasNet was pretrained on AbdomenAtlas, which is assembled from public datasets that include these four, so the test set is held out from our fine-tuning but not necessarily from pretraining.

## Labels in each dataset

| Structure | TotalSegmentator v2 | AMOS22 id | BTCV id | FLARE22 id |
|---|---|---|---|---|
| spleen | spleen | 1 | 1 | 3 |
| kidney right | kidney_right + kidney_cyst_right | 2 | 2 | 2 |
| kidney left | kidney_left + kidney_cyst_left | 3 | 3 | 13 |
| liver | liver | 6 | 6 | 1 |
| aorta | aorta | 8 | 8 | 5 |
| inferior vena cava | inferior_vena_cava | 9 | 9 | 6 |
| pancreas | pancreas | 10 | 11 | 4 |
| adrenal right | adrenal_gland_right | 11 | 12 | 7 |
| adrenal left | adrenal_gland_left | 12 | 13 | 8 |

All other structures become background. Structures present in only two of the three datasets (portal/splenic vein, duodenum, bladder, prostate/uterus) cannot be used without partial-label handling and are left out. `cohort_scan.py` logs the median volume of each organ per dataset, which exposes a wrong id immediately.

Known differences between sources: TotalSegmentator labels the aorta and vena cava wherever they are in view, including the chest, while AMOS and BTCV are abdominal; TotalSegmentator is distributed at 1.5 mm, so its 1.0 mm version is upsampled.

## Notes on resampling and reading

The three resamplers available (nnU-Net's default, AtlasNet's `resample_torch_fornnunet`, SimpleITK) do not produce the same image: on a test volume the SimpleITK and default CT results differ by about 8% of the intensity standard deviation on average. So the same one must prepare the training cache and the images at inference; SimpleITK was chosen because deployment uses it. It has no separate handling of thick slices (one 3D resample). The SimpleITK logits resampler reproduces nnU-Net's default to float precision (max difference 5e-7, no label changes).

About one in seven TotalSegmentator CTs in a local sample has an orientation matrix skewed by about 1e-4. ITK-based readers (nnU-Net's default `SimpleITKIO`) refuse these files; the nibabel reader opens them. `dataset.json` therefore names `NibabelIOWithReorient`. The CT files themselves are not altered.

### Note on the CT intensity window

Decided: keep AtlasNet's [-1000, 629] HU. The window is wide because AtlasNet's foreground included lungs and air; nnU-Net's own rule applied to our nine organs would give roughly [-55, 352] HU (estimate from 6 local scans). Reasons to keep it: the transferred weights were trained with it; nnU-Net's plan transfer carries it over; fat around the adrenal (about -100 HU) stays inside the window, whereas [-55, 352] would clip it; and the first convolution is followed by instance normalization, which removes any overall rescaling of the input. The effect of a narrower window was not tested.

## Open

1. Redistribution terms of BTCV and of FLARE22 (research-only labels), before any files derived from them are made public.
2. Confirm the nnU-Net plans for the new dataset keep AtlasNet's architecture so its weights load.

## Work items

1. Run `colab_prepare.sh` on Colab: final inclusion table, the cleaned dataset and the split.
2. Run `preprocess.py` on Colab. Written as: transfer AtlasNet's plans with nnU-Net's `move_plans_between_datasets`, set the SimpleITK resamplers, copy `dataset.json` beside the plans, run nnU-Net's preprocessing on Colab, publish the cache to Hugging Face. (Tried locally on 5 scans: works; the transfer changes only the dataset name, plans name and data identifier; adrenal volumes in the 1 mm cache are within 1.5% of the originals.)
3. On Google Cloud (`gcp/create_vm.sh`): a short timing test first, then the full run; `predict.py` follows automatically. The Google Cloud scripts and the pinned environment have not been run yet.
4. Add surface Dice and signed/absolute volume error to the evaluation (`predict.py` currently reports nnU-Net's Dice and voxel counts).
