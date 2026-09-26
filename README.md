# adrenal-multiorgan

Fine-tuning AtlasNet (nnU-Net) to segment the adrenal glands, kidneys, liver, spleen, aorta, inferior vena cava and pancreas on CT, from four public datasets: TotalSegmentator v2, AMOS22 CT, BTCV and FLARE22. The full specification and the reasons behind each choice are in [PLAN.md](PLAN.md).

**Continuing this work (agents and people): read [AGENTS.md](AGENTS.md) first** for current state, VM access and open decisions.

D997/D998 historical training and validation membership, recovered case lists, and source evidence are in [docs/MODEL_SPLITS.md](docs/MODEL_SPLITS.md). D997 has a corroborated 1,082-case training list and 57 verified validation IDs; D998 has 56 verified validation IDs but its exact training list remains unknown.

Recovered AbdomenAtlas-to-AMOS/BTCV/FLARE case mappings and matching-method limitations are in [docs/ABDOMENATLAS_PROVENANCE.md](docs/ABDOMENATLAS_PROVENANCE.md).

The work is a short sequence of steps. Each step is one small script that logs what it did and records the exact inputs it used. Everything runs on one Google Cloud machine (one NVIDIA RTX PRO 6000, 48 vCPUs), started with one command; each script can also be run by hand.

| Step | Script | What it does | Status (September 19, 2026) |
|---|---|---|---|
| 1 | `cohort_scan.py` | Measures every scan and applies the label rules: adrenal size, fragments, glands cut by the scan edge, left/right check, slice thickness | ran on Google Cloud on all 1,608 scans, no read errors |
| 2 | `build_dataset.py` | Writes the cleaned 9-class nnU-Net dataset and the stratified test/validation/training split | ran on Google Cloud: 1,394 scans written, none failed |
| 3 | `preprocess.py` | nnU-Net's integrity check and preprocessing to 1.0 mm with AtlasNet's plans and SimpleITK resampling; continues after an interruption | ran to completion on Google Cloud (about 20 minutes, 170 GB); the resumable version is tested locally only |
| 1-3 | `prepare.sh` | Downloads the four datasets from Hugging Face and runs steps 1-3; uploads the tables of steps 1-2 before preprocessing starts | ran to completion on Google Cloud |
| cache | `hf_cache.py` | Restores a verified copy of the prepared data on a new machine, or uploads one in the background while training runs; atomic checkpoint snapshots | 27 offline tests pass; no real upload or restore has completed yet |
| 4 | `train.py`, `trainers/adrenal_multiorgan.py` | Training: one command, continues by itself | trainer checked on CPU; no epoch has run on a GPU yet |
| 5 | `predict.py` | Segments the held-out test scans and scores them with `evaluate.py` | scoring tested; prediction not yet run |
| 6 | `evaluate.py` | Dice, surface Dice (0.5, 0.8, 1.0 mm; `surface-distance` package) and volume error per scan and organ; reports for all organs, adrenals, aorta | tested on synthetic masks |
| – | `compare.py`, `batched_predictor.py`, `colab_compare.sh` | The same prediction and scoring for other models (AtlasNet, two TotalSegmentator-label models) on a Colab A100 | not yet run |
| all | `gcp/` | Creates or restarts the Spot machine, which runs everything above and switches itself off | creation, setup and preparation work; training and the automatic switch-off are not yet confirmed |

`vendor/nnUNet` is nnU-Net 2.8.1 plus two SimpleITK resampling modules; see [vendor/README.md](vendor/README.md). `pixi.lock` pins the environment of the machine.

## Running it on Google Cloud

In a terminal with the Google Cloud CLI (or Google Cloud Shell: console.cloud.google.com, the `>_` button), with the project selected:

```bash
git clone https://github.com/ahigazy1/adrenal-multiorgan
bash adrenal-multiorgan/gcp/create_vm.sh
```

This creates a Spot machine with one RTX PRO 6000 in the first us-central1 zone that has room; Google stops it after at most 100 hours of running in one go. A Hugging Face token must be in the project's Secret Manager under the name `HF_TOKEN`; if it is not there yet, the script asks for it once. **The token needs read and write access to both `ahigazy1/adrenal-multiorgan-cache` and `ahigazy1/adrenal-multiorgan-model`.** The machine then works through everything by itself and switches itself off at the end:

1. installs the pinned environment (about a minute);
2. `hf_cache.py ensure`: restores a complete, compatible copy of the prepared data from `ahigazy1/adrenal-multiorgan-cache`, or, if there is none, runs `prepare.sh` (downloads, scan, dataset, split, preprocessing; about 40 minutes). The tables of the scan and the split appear at `https://huggingface.co/ahigazy1/adrenal-multiorgan-model/tree/main/dataset` before preprocessing starts;
3. starts uploading the prepared data (about 170 GB) **in the background**, at the lowest CPU and disk priority. Training does not wait for it; it is only needed by a future replacement machine (log: `/var/log/adrenal-cache-upload.log`);
4. `train.py`: restores a compatible checkpoint when there is one, then continues toward 2000 epochs, with atomic checkpoint/log snapshots uploaded every 100 epochs;
5. `predict.py`: the held-out test scans; then it waits for the background upload if that is still running.

The prepared data contains the `.b2nd` arrays, case properties, patch-sampling index, plans, splits, validation labels, test scans and AtlasNet weights. A completion manifest is uploaded last; a partial upload never counts as ready. See [docs/HF_CACHE.md](docs/HF_CACHE.md) for the integrity, compatibility and recovery rules.

Google may stop a Spot machine at any time. **Running the same line again starts it again**: preparation that was finished is kept (an interrupted preprocessing continues with the scans that are left), training continues from the last checkpoint on the disk, and the background upload resumes. `bash adrenal-multiorgan/gcp/progress.sh` shows the state and the latest lines, also after the machine has switched off. When the work is finished, `bash adrenal-multiorgan/gcp/delete_vm.sh` deletes the machine and its disk; the disk costs money for as long as it exists. A replacement machine can restore only what finished uploading: the prepared data if its upload completed (otherwise it prepares again, about 40 minutes), and the last uploaded checkpoint.

To follow the work from inside the machine (`gcloud compute ssh adrenal-train --zone=us-central1-f`; no `sudo` needed):

```bash
tail -n 40 -f /var/log/adrenal.log
```

During preprocessing that log is quiet on purpose; the progress bar is in `/opt/adrenal-multiorgan/data/nnUNet_preprocessed/Dataset902_AdrenalMultiorgan/preprocess-console.log`. During training, nnU-Net's own log with the losses and the Dice of every organ is `/opt/adrenal-multiorgan/data/nnUNet_results/*/*/fold_0/training_log_*.txt`.

## The steps by hand

On any Linux machine with an NVIDIA GPU and about 350 GB of disk, after `pixi install --locked` and `hf auth login`:

```bash
pixi run python hf_cache.py ensure --data data
pixi run python train.py
pixi run python predict.py
```

`hf_cache.py ensure` does not upload anything; `pixi run python hf_cache.py publish --data data` uploads the prepared data and can run beside training. `pixi run bash prepare.sh data` runs the preparation alone.

`cohort_scan.py` changes no data. It writes `results/cohort_scan.csv` (one row per case: organ volumes in mL, each gland's connected components and decision, the left/right check, and the outcome `kept_with_adrenal`, `kept_no_adrenal` or `excluded`), `cohort_scan.json` (input checksums, package versions, thresholds, counts) and a log that includes the median size of each organ per dataset, where a wrong label id would show at once.

`build_dataset.py` writes `nnUNet_raw/Dataset902_AdrenalMultiorgan`: every CT that the scan kept (bytes copied unchanged), one 9-class label file per CT with adrenal speckles removed and the CT's own orientation header, `dataset.json`, the split, and `build_dataset.csv` (role, speckle volume removed, overlapping voxels, labels present). Each gland is judged again with the scan's rule and must get the scan's decision. One fifth of every source is held out as the test set and the rest is divided into nnU-Net's five folds, both with scikit-learn's `StratifiedKFold`, so that each part has the same mix of sources, slice thicknesses and adrenal volumes (see PLAN.md).

`preprocess.py` runs nnU-Net's dataset integrity check, transfers AtlasNet's plans with nnU-Net's `move_plans_between_datasets`, sets the resampling functions to SimpleITK, runs nnU-Net's per-case preprocessing function on every scan that is not finished yet, builds nnU-Net's patch-sampling index, and downloads and verifies the AtlasNet checkpoint.

`train.py` starts, continues after an interruption, or restores a verified checkpoint snapshot from Hugging Face. It checks the trainer, plans, split and cached data identity before resuming. All settings that differ from stock nnU-Net are listed at the top of `trainers/adrenal_multiorgan.py`; `python check_trainer.py atlasnet_plans.json` confirms them without a GPU or data (`atlasnet_plans.json` is the plans file shipped with the AtlasNet weights).

`predict.py` uses nnU-Net's predictor without mirroring (`--checkpoint best` for the best instead of the final checkpoint) and `evaluate.py`: Dice, surface Dice at 0.5, 0.8 and 1.0 mm and volume error per scan and organ in `data/predictions/<checkpoint>/evaluation/`.

To try steps 1 and 2 on a laptop (`pip install -r requirements.txt`):

```bash
python cohort_scan.py --out test-results --limit 40 --ts /path/to/Totalsegmentator_dataset_v201.zip
python build_dataset.py --scan test-results/cohort_scan.csv --out test-results/data --ts /path/to/Totalsegmentator_dataset_v201.zip
```

## Data sources and licences

- TotalSegmentator v2.0.1: https://zenodo.org/records/10047292 (CC BY 4.0). The machine downloads our private Hugging Face copy of the same zip, because Zenodo is slow, and checks it against the checksum Zenodo publishes
- AMOS22: https://huggingface.co/datasets/MedOtter/amos22-ct-dataset (CC BY 4.0)
- BTCV: https://huggingface.co/datasets/lingheng123/btcv (originally distributed through Synapse under its own terms)
- FLARE22: https://huggingface.co/datasets/MedOtter/FLARE22 (images CC BY-SA 4.0, labels for research use; https://zenodo.org/records/7860267)
- AtlasNet weights: https://huggingface.co/AbdomenAtlas/AtlasNet
