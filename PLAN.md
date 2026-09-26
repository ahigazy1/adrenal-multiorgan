# Specification

Historical specification for the completed Dataset902 run (September 19, 2026). The run finished 2,000 epochs; the evaluated best checkpoint is epoch 1957. Current operations and the separate, paused D997 proposal are in [AGENTS.md](AGENTS.md). The design below records Dataset902, not approval for another run.

## Setup

Everything runs on one Google Cloud Spot VM with an NVIDIA RTX PRO 6000 (96 GB) and 48 vCPUs: download, label cleaning, split, nnU-Net preprocessing, training and prediction on the held-out scans. It is started with one command and may be interrupted at any time, so every stage either continues or is skipped when the machine is started again. The tables of the cleaning and the split, the logs, the checkpoints and the predictions' scores are uploaded to Hugging Face. The prepared data (about 170 GB) is uploaded as well, in the background while training runs, so that a replacement machine can restore it instead of preparing again.

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
| Hugging Face | private model repository `ahigazy1/adrenal-multiorgan-model`: tables, logs, checkpoints. Private dataset `ahigazy1/adrenal-multiorgan-cache`: our copy of the TotalSegmentator zip (Zenodo is slow) and the prepared data as about 10 GiB tar parts |
| Machine limits | Spot pricing, us-central1, stopped by Google after at most 100 hours of running in one go; 500 GB disk |
| Checkpoints | every 100 epochs, plus the best (by nnU-Net's validation Dice moving average); uploaded to Hugging Face every 100 epochs |

### Adrenal label cleaning rule

Applied per gland, to every source, in training and test data. Connected components are 26-connected on the original label grid; volumes use the original voxel size.

1. Gland absent: the scan is kept.
2. Secondary components smaller than 0.3 mL are speckles and are removed.
3. A secondary component of 0.3 mL or more means the label is broken: the whole scan is excluded.
4. The remaining component must be larger than 1 mL, otherwise the whole scan is excluded.
5. A gland that touches any edge of the scan volume is cut off, so its volume is not the gland's volume: the whole scan is excluded.
6. Left/right check, in world coordinates: the left adrenal and left kidney must lie to the patient's left of the right ones, and the spleen to the left of the liver (only organs that are present are compared). A scan that fails is excluded. This catches swapped labels, a wrong orientation in the file header and mirrored anatomy.
7. Slice thickness (the coarsest voxel axis) over 5 mm: the scan is excluded.
8. Every removal and exclusion is recorded per case.

Excluding the whole scan (rather than masking one gland) keeps the rule one sentence long. "Touches any edge" (rule 5) means any of the six faces of the image volume, not only the first and last slice; this conservative reading was kept deliberately.

### What the rule excluded (Google Cloud run, September 19, 2026)

1,608 scans were read without error; 214 were excluded and 1,394 kept.

| Source | Scanned | Kept with adrenals | Kept without adrenals | Excluded | Reasons (scans) |
|---|---:|---:|---:|---:|---|
| TotalSegmentator | 1,228 | 626 | 396 | 206 (16.8%) | gland cut by an edge 111; gland of 1 mL or less 68; both 23; fragmented 4 |
| AMOS22 | 300 | 292 | 0 | 8 (2.7%) | small 4; fragmented 2; cut by an edge 1; left/right check and small 1 (`amos_0156`) |
| BTCV | 30 | 30 | 0 | 0 | |
| FLARE22 | 50 | 50 | 0 | 0 | |

Per rule (a scan can break more than one): cut by an edge 135 scans (190 glands, median 3.1 mL, so ordinary glands at the border of tightly cropped scans); 1 mL or less 96 scans (118 glands, median 0.35 mL); fragmented 6; left/right check 1. No scan in any source has a slice thickness over 5 mm, so rule 7 excluded nothing. Median organ volumes were plausible in every source (liver 1,221-1,660 mL, kidneys 128-196 mL, adrenals 3.1-5.5 mL), which confirms the label ids below.

### Learning-rate schedule: why this one

It is nnU-Net's own fine-tuning recipe, unchanged (`PretrainedTrainer` and `nnUNetTrainer_warmup` in the vendored code: SGD, peak 1e-3 when pretrained weights are loaded versus 1e-2 from scratch, 50 warmup epochs, offset polynomial decay). Not chosen: a higher peak (in preliminary 100-epoch runs 5e-3 was ahead of 1e-3 only within single-run noise, and the reason for it, too few updates, does not apply to 2000 epochs); scaling the rate with batch size (nnU-Net never does); cosine schedules, restarts, layer-wise rates or freezing (not standard in nnU-Net, no evidence they are needed).

### Held-out test set and training split

No dataset's published split is used. All kept scans are divided by one rule, so every source contributes the same share and the parts have the same mix:

- Scans are grouped by source, slice thickness (<=2 mm, 2-4 mm, 4-5 mm; the coarsest voxel axis) and total adrenal volume (none, or the lower, middle or upper third of scans that have adrenals). Groups with fewer than 10 scans fall back to source and volume, then to source alone.
- scikit-learn's `StratifiedKFold` (5 folds, shuffled, seed 12345) divides every group evenly. One fifth of all kept scans is the held-out test set.
- The remaining four fifths are divided the same way into five folds for nnU-Net; fold 0 is trained, so its validation part is one fifth of the remainder (16% of all scans) and its training part is 64%.
- `build_dataset.py` logs and saves the size, adrenal-volume median, slice-thickness mix and per-group counts of each part.

Result of the split on the Google Cloud run:

| Part | Scans | With adrenals | Median total adrenal volume | Slices <=2 / 2-4 / 4-5 mm |
|---|---:|---:|---:|---|
| Test | 279 (204 TotalSegmentator, 59 AMOS22, 6 BTCV, 10 FLARE22) | 200 (71.7%) | 7.59 mL | 218 / 15 / 46 |
| Validation (fold 0) | 223 | 160 (71.7%) | 7.95 mL | 174 / 12 / 37 |
| Training (fold 0) | 892 | 638 (71.5%) | 7.66 mL | 700 / 46 / 146 |

A few scans whose own group is too small fall back to a coarser group label and then form tiny groups of their own, which `StratifiedKFold` places freely (it warns about this). The balance above shows this had no visible effect. The rule must not be changed once a run uses this split, because that would change which scans are held out.

TotalSegmentator is distributed at 1.5 mm isotropic, so slice-thickness variety comes from AMOS22, BTCV and FLARE22 only. Because published test splits are not used, results are not directly comparable with numbers other papers report on those splits. Limitation that stays: AtlasNet was pretrained on AbdomenAtlas, which is assembled from public datasets that include these four, so the test set is held out from our fine-tuning but not necessarily from pretraining.

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
2. Label resampling uses SimpleITK's label-linear interpolation. It makes small structures slightly smaller at 1 mm (about 1-2% for adrenals in local checks); to be stated in the methods.

## Work items

1. First epochs on Google Cloud: time per epoch and GPU memory decide whether 2000 epochs fit the 100-hour limit of one run (about 71 hours at 128 s per epoch).
2. Confirm on the real machine: the background upload of the prepared data, a restore on a replacement machine, the automatic switch-off, and `predict.py`.
3. Run `compare.py` (our model, AtlasNet and two TotalSegmentator-label models on the test scans; `evaluate.py` reports Dice, surface Dice and volume error).

## Lessons from the first runs

- nnU-Net's progress bar writes carriage returns without newlines; sent through Google's startup-script logger it overflowed the logger, which closed the pipe and killed preprocessing. The bar now goes to a file.
- Downloading the four sources at the same time got "429 Too Many Requests" from Hugging Face's CDN; they are downloaded one after the other, with retries. One after the other takes about 6 minutes.
- nnU-Net's `preprocess_dataset()` imports `distutils`, which Python 3.12 removed; its three parts are called directly.
- About one in seven TotalSegmentator CTs has a slightly skewed orientation matrix that ITK-based readers refuse; the nibabel reader is used.
- Measured on the machine: scan about 6 minutes, dataset about 6 minutes, preprocessing about 20 minutes with 24-32 workers, prepared data 170 GB.
