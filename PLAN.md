# Specification

Last updated September 19, 2026. Status: steps 1, 2, 4 and 5 are written and checked on CPU; nothing has been run on Colab or a GPU yet. Results used for publication come only from logged Colab and GPU runs. Local runs are for testing code.

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
| Train/validation split | nnU-Net's default (5 folds, seed 12345) over all non-test scans, fold 0; validation is used for monitoring and for choosing the best checkpoint |
| Checkpoints | every 100 epochs, plus the best (by nnU-Net's validation Dice moving average); uploaded to Hugging Face every 100 epochs |

### Adrenal label cleaning rule

Applied per gland, to every source, in training and test data. Connected components are 26-connected on the original label grid; volumes use the original voxel size.

1. Gland absent: the scan is kept.
2. Secondary components smaller than 0.3 mL are speckles and are removed.
3. A secondary component of 0.3 mL or more means the label is broken: the whole scan is excluded.
4. The remaining component must be larger than 1 mL, otherwise the whole scan is excluded (typically a gland cut by the edge of the scan).
5. Every removal and exclusion is recorded per case.

Excluding the whole scan (rather than masking one gland) keeps the rule one sentence long. A local test on TotalSegmentator suggests it costs about 8% of that dataset.

### Learning-rate schedule: why this one

It is nnU-Net's own fine-tuning recipe, unchanged (`PretrainedTrainer` and `nnUNetTrainer_warmup` in the vendored code: SGD, peak 1e-3 when pretrained weights are loaded versus 1e-2 from scratch, 50 warmup epochs, offset polynomial decay). Not chosen: a higher peak (in preliminary 100-epoch runs 5e-3 was ahead of 1e-3 only within single-run noise, and the reason for it, too few updates, does not apply to 2000 epochs); scaling the rate with batch size (nnU-Net never does); cosine schedules, restarts, layer-wise rates or freezing (not standard in nnU-Net, no evidence they are needed).

### Held-out test set

Each dataset's official split is used where one exists, so the choice is not ours:

- TotalSegmentator: the 89 `test` cases in its `meta.csv`; its 1082 train and 57 val cases are used for training.
- AMOS22 CT: the official validation set (100 scans) is our test set; the 200 official training scans are used for training. AMOS test labels are not public.
- BTCV and FLARE22: no labelled test set is public, so a seeded random 20% of each (after the cleaning rule) is held out: about 6 and 10 scans.

Adrenal metrics are reported on the test scans that contain adrenals after the cleaning rule. Limitation: AtlasNet was pretrained on AbdomenAtlas, which is assembled from public datasets that include these three, so the test set is held out from our fine-tuning but not necessarily from pretraining.

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

AtlasNet's window is wide because its foreground included lungs and air. nnU-Net's own rule applied to our nine organs would give roughly [-55, 352] HU (estimate from 6 local cases). Arguments for keeping AtlasNet's: the transferred weights were trained with it; it is what nnU-Net's plan transfer does; fat around the adrenal (about -100 HU) stays inside the window, whereas a [-55, 352] window would clip it and reduce the fat-to-gland contrast; and the network's first layer is followed by instance normalization, which removes any overall rescaling of the input, so a wide window does not by itself weaken soft-tissue contrast. Argument for narrowing: values outside the abdomen's range carry no information for these organs. Changing it means the pretrained first layers see differently clipped images and must re-adapt. This is untested here; if it matters it should be settled by a short paired experiment, not by argument.

## Open

1. Redistribution terms of BTCV and of FLARE22 (research-only labels), before any files derived from them are made public.
3. CT intensity window: keep AtlasNet's [-1000, 629] HU (current choice) or train with a narrower soft-tissue window. See the note below.
2. Confirm the nnU-Net plans for the new dataset keep AtlasNet's architecture so its weights load.

## Work items

1. Run `colab_prepare.sh` on Colab: final inclusion table, the cleaned dataset and the split.
2. Step 3 script: transfer AtlasNet's plans with nnU-Net's `move_plans_between_datasets`, set the SimpleITK resamplers, copy `dataset.json` beside the plans, run nnU-Net's preprocessing on Colab, publish the cache to Hugging Face. (Tried locally on 5 scans: works; the transfer changes only the dataset name, plans name and data identifier; adrenal volumes in the 1 mm cache are within 1.5% of the originals.)
3. Five-epoch timing test of `train.py` on the GPU VM, then the full run, then `predict.py`.
4. Add surface Dice and signed/absolute volume error to the evaluation (`predict.py` currently reports nnU-Net's Dice and voxel counts).
