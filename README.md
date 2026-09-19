# adrenal-multiorgan

Fine-tuning AtlasNet (nnU-Net) to segment the adrenal glands, kidneys, liver, spleen, aorta, inferior vena cava and pancreas on CT, from four public datasets: TotalSegmentator v2, AMOS22 CT, BTCV and FLARE22. The full specification and the reasons behind each choice are in [PLAN.md](PLAN.md).

The work is a short sequence of steps. Each step is one small script that logs what it did and records the exact inputs it used. Everything runs on one Google Cloud machine (one NVIDIA RTX PRO 6000, 48 vCPUs), started with one command; each script can also be run by hand.

| Step | Script | What it does | Status |
|---|---|---|---|
| 1 | `cohort_scan.py` | Measures every scan and applies the label rules: adrenal size, fragments, glands cut by the scan edge, left/right check | tested locally on TotalSegmentator |
| 2 | `build_dataset.py` | Writes the cleaned 9-class nnU-Net dataset and the stratified test/validation/training split | tested locally on TotalSegmentator |
| 3 | `preprocess.py` | nnU-Net's integrity check and preprocessing to 1.0 mm with AtlasNet's plans and SimpleITK resampling | tested locally on a few scans |
| 1-3 | `prepare.sh` | Downloads the four datasets and runs steps 1-3; uploads the tables of steps 1-2 before preprocessing starts | not yet run |
| 4 | `train.py`, `trainers/adrenal_multiorgan.py` | Training: one command, continues by itself | trainer checked on CPU; not yet run on a GPU |
| 5 | `predict.py` | Segments the held-out test scans and scores them with nnU-Net's evaluator | evaluator half tested; prediction not yet run |
| all | `gcp/` | Creates or restarts the Spot machine, which runs everything above and switches itself off | not yet run |

`vendor/nnUNet` is nnU-Net 2.8.1 plus two SimpleITK resampling modules; see [vendor/README.md](vendor/README.md). `pixi.lock` pins the environment of the machine.

## Running it on Google Cloud

In a terminal with the Google Cloud CLI (or Google Cloud Shell: console.cloud.google.com, the `>_` button), with the project selected:

```bash
git clone https://github.com/ahigazy1/adrenal-multiorgan
bash adrenal-multiorgan/gcp/create_vm.sh
```

This creates a Spot machine with one RTX PRO 6000 in the first zone that has room. A Hugging Face token must be in the project's Secret Manager under the name `HF_TOKEN`; if it is not there yet, the script asks for it once. The machine then works through everything by itself and switches itself off at the end:

1. installs the pinned environment;
2. `prepare.sh`: downloads the datasets, scans them, builds the dataset and split, uploads those tables to `https://huggingface.co/ahigazy1/adrenal-multiorgan-model/tree/main/dataset` (look at the exclusions and the split here while preprocessing runs), preprocesses; a few hours in total;
3. `train.py`: 2000 epochs, with the latest and best checkpoints and the logs uploaded every 100 epochs;
4. `predict.py`: the held-out test scans.

Google may stop a Spot machine at any time. **Running the same line again starts it again**, and it continues where it was (training from the last checkpoint). `bash adrenal-multiorgan/gcp/progress.sh` shows the state and the latest lines, also after the machine has switched off. When the work is finished, `bash adrenal-multiorgan/gcp/delete_vm.sh` deletes the machine and its disk; the disk costs money for as long as it exists.

## The steps by hand

On any Linux machine with an NVIDIA GPU and about 350 GB of disk, after `pixi install --locked` and `hf auth login`:

```bash
pixi run bash prepare.sh data
pixi run python train.py
pixi run python predict.py
```

`cohort_scan.py` changes no data. It writes `results/cohort_scan.csv` (one row per case: organ volumes in mL, each gland's connected components and decision, the left/right check, and the outcome `kept_with_adrenal`, `kept_no_adrenal` or `excluded`), `cohort_scan.json` (input checksums, package versions, thresholds, counts) and a log that includes the median size of each organ per dataset, where a wrong label id would show at once.

`build_dataset.py` writes `nnUNet_raw/Dataset902_AdrenalMultiorgan`: every CT that the scan kept (bytes copied unchanged), one 9-class label file per CT with adrenal speckles removed and the CT's own orientation header, `dataset.json`, the split, and `build_dataset.csv` (role, speckle volume removed, overlapping voxels, labels present). Each gland is judged again with the scan's rule and must get the scan's decision. One fifth of every source is held out as the test set and the rest is divided into nnU-Net's five folds, both with scikit-learn's `StratifiedKFold`, so that each part has the same mix of sources, slice thicknesses and adrenal volumes (see PLAN.md).

`preprocess.py` runs nnU-Net's dataset integrity check, transfers AtlasNet's plans with nnU-Net's `move_plans_between_datasets`, sets the resampling functions to SimpleITK, runs nnU-Net's preprocessor, and downloads and verifies the AtlasNet checkpoint.

`train.py` starts, continues after an interruption, or restores the latest checkpoint from Hugging Face. All settings that differ from stock nnU-Net are listed at the top of `trainers/adrenal_multiorgan.py`; `python check_trainer.py atlasnet_plans.json` confirms them without a GPU or data (`atlasnet_plans.json` is the plans file shipped with the AtlasNet weights).

`predict.py` uses nnU-Net's predictor without mirroring (`--checkpoint best` for the best instead of the final checkpoint) and nnU-Net's evaluator: Dice and voxel counts per scan and label in `data/predictions/<checkpoint>/summary.json`.

To try steps 1 and 2 on a laptop (`pip install -r requirements.txt`):

```bash
python cohort_scan.py --out test-results --limit 40 --ts /path/to/Totalsegmentator_dataset_v201.zip
python build_dataset.py --scan test-results/cohort_scan.csv --out test-results/data --ts /path/to/Totalsegmentator_dataset_v201.zip
```

## Data sources and licences

- TotalSegmentator v2.0.1: https://zenodo.org/records/10047292 (CC BY 4.0)
- AMOS22: https://huggingface.co/datasets/MedOtter/amos22-ct-dataset (CC BY 4.0)
- BTCV: https://huggingface.co/datasets/lingheng123/btcv (originally distributed through Synapse under its own terms)
- FLARE22: https://huggingface.co/datasets/MedOtter/FLARE22 (images CC BY-SA 4.0, labels for research use; https://zenodo.org/records/7860267)
- AtlasNet weights: https://huggingface.co/AbdomenAtlas/AtlasNet
