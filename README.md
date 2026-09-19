# adrenal-multiorgan

Fine-tuning AtlasNet (nnU-Net) to segment the adrenal glands, kidneys, liver, spleen, aorta, inferior vena cava and pancreas on CT, from four public datasets: TotalSegmentator v2, AMOS22 CT, BTCV and FLARE22. The full specification and the reasons behind each choice are in [PLAN.md](PLAN.md).

The work is a short sequence of steps. Each step is one small script that logs what it did, records the exact inputs it used, and changes nothing outside its output folder. Data preparation runs on a Google Colab runtime; training runs on a single GPU VM.

| Step | Script | What it does | Status |
|---|---|---|---|
| 1 | `cohort_scan.py` | Measures every scan and applies the adrenal label cleaning rule | tested locally, not yet run on Colab |
| 2 | `build_dataset.py` | Writes the cleaned 9-class nnU-Net dataset and the train/validation/test split | tested locally, not yet run on Colab |
| 3 | `preprocess.py` | nnU-Net preprocessing to 1.0 mm with AtlasNet's plans, published to Hugging Face in groups | tested locally without uploading, not yet run on Colab |
| 4 | `train.py`, `trainers/adrenal_multiorgan.py`, `gcp/` | Training on a Google Cloud Spot VM: one command creates or restarts it, training continues by itself | trainer checked on CPU; nothing run on a GPU or on Google Cloud yet |
| 5 | `predict.py` | Segments the held-out test scans and scores them with nnU-Net's evaluator | evaluator half tested, prediction not yet run on a GPU |

`vendor/nnUNet` is nnU-Net 2.8.1 plus two SimpleITK resampling modules; see [vendor/README.md](vendor/README.md).

## Steps 1 and 2: scan the sources and build the dataset

On a Colab runtime (a TPU runtime is convenient because it has many CPUs; the TPU itself is not used):

```bash
git clone https://github.com/ahigazy1/adrenal-multiorgan && cd adrenal-multiorgan
bash colab_prepare.sh
```

This downloads TotalSegmentator v2.0.1 from Zenodo (23.6 GB, checksum verified) and AMOS22 CT, BTCV and FLARE22 from Hugging Face at pinned revisions (about 80 GB of disk in total), then runs the two scripts.

`cohort_scan.py` changes no data. It writes to `results/`:

- `cohort_scan.csv`: one row per case with organ volumes (mL), each gland's connected components and decision, and the scan-level outcome (`kept_with_adrenal`, `kept_no_adrenal`, `excluded`).
- `cohort_scan.json`: input checksums, package versions, rule thresholds, outcome counts, and the CSV's checksum.
- `cohort_scan.log`: everything printed, including the median size of each organ per dataset (a wrong label id shows up there at once).

`build_dataset.py` writes `nnUNet_raw/Dataset902_AdrenalMultiorgan`: every CT that the scan kept (bytes copied unchanged), one 9-class label file per CT with adrenal speckles removed, `dataset.json`, the split, and `build_dataset.csv` with one row per case (role, speckle volume removed, overlapping voxels, labels present). Each gland is judged again with the scan's rule and must get the scan's decision, otherwise the case fails loudly. One fifth of every source is held out as the test set and the rest is divided into nnU-Net's five folds, both with scikit-learn's `StratifiedKFold` so that each part has the same mix of sources, slice thicknesses and adrenal volumes (see PLAN.md). The mix of each part is logged.

To try both locally on a few cases:

```bash
pip install -r requirements.txt
python cohort_scan.py --out test-results --limit 40 --ts /path/to/Totalsegmentator_dataset_v201.zip
python build_dataset.py --scan test-results/cohort_scan.csv --out test-results/data --ts /path/to/Totalsegmentator_dataset_v201.zip
```

Both scripts run a small self-check of their rules every time they start.

## Step 3: preprocess and publish the cache

On the same Colab runtime, after looking at the tables from steps 1 and 2:

```bash
pip install -e vendor/nnUNet
hf auth login
python preprocess.py --data /content/data
```

AtlasNet's plans are transferred with nnU-Net's own `move_plans_between_datasets`, the resampling functions are set to SimpleITK, and every training scan is preprocessed with nnU-Net's own per-case function. The cache is about 200 GB, more than a Colab disk holds, so scans are processed 100 at a time, uploaded to the Hugging Face dataset `ahigazy1/adrenal-multiorgan-cache` and deleted locally. Scans already uploaded are skipped, so after an interruption the same command continues. The plans, the split, the labels used for validation, the held-out test scans and the AtlasNet checkpoint are uploaded too: the training machine needs only this one repository.

Zenodo serves TotalSegmentator slowly. `colab_prepare.sh` first looks for the same zip in that repository (checked against Zenodo's published checksum either way); to put it there once: `hf upload ahigazy1/adrenal-multiorgan-cache Totalsegmentator_dataset_v201.zip sources/Totalsegmentator_dataset_v201.zip --repo-type dataset`.

## Steps 4 and 5: training and prediction on Google Cloud

In Google Cloud Shell (console.cloud.google.com, the `>_` button; nothing to install), with the project selected:

```bash
git clone https://github.com/ahigazy1/adrenal-multiorgan
bash adrenal-multiorgan/gcp/create_vm.sh
```

The first time, this asks once for a Hugging Face token (stored in Secret Manager), then creates a Spot machine with one RTX PRO 6000 in the first zone that has room. The machine installs the pinned environment (`pixi.lock`), downloads the cache, trains, segments the held-out test scans, uploads everything and switches itself off. Google may stop a Spot machine at any time; **running the same line again starts it again**, and training continues from the last checkpoint. `bash adrenal-multiorgan/gcp/progress.sh` shows the state and the latest epochs. When the work is finished, `bash adrenal-multiorgan/gcp/delete_vm.sh` deletes the machine and its disk; the disk costs money for as long as it exists.

What runs on the machine is two commands, which can also be used by hand on any Linux machine with an NVIDIA GPU (`pixi install --locked` first):

```bash
pixi run python train.py
pixi run python predict.py
```

`train.py` starts, continues after an interruption, or restores the latest checkpoint from Hugging Face onto a new machine. Every 100 epochs the latest and best checkpoints and the logs are uploaded. All settings that differ from stock nnU-Net are listed at the top of `trainers/adrenal_multiorgan.py`. To confirm them without a GPU or data: `python check_trainer.py atlasnet_plans.json` (`atlasnet_plans.json` is the plans file shipped with the AtlasNet weights).

`predict.py` segments the held-out test scans with the final checkpoint (`--checkpoint best` for the best one) using nnU-Net's predictor without mirroring, and scores them with nnU-Net's evaluator: Dice and voxel counts per scan and label in `data/predictions/<checkpoint>/summary.json`.

## Data sources and licences

- TotalSegmentator v2.0.1: https://zenodo.org/records/10047292 (CC BY 4.0)
- AMOS22: https://huggingface.co/datasets/MedOtter/amos22-ct-dataset (CC BY 4.0)
- BTCV: https://huggingface.co/datasets/lingheng123/btcv (originally distributed through Synapse under its own terms)
- FLARE22: https://huggingface.co/datasets/MedOtter/FLARE22 (images CC BY-SA 4.0, labels for research use; https://zenodo.org/records/7860267)
- AtlasNet weights: https://huggingface.co/AbdomenAtlas/AtlasNet
