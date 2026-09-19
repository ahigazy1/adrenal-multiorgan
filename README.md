# adrenal-multiorgan

Fine-tuning AtlasNet (nnU-Net) to segment the adrenal glands, kidneys, liver, spleen, aorta, inferior vena cava and pancreas on CT, from three public datasets: TotalSegmentator v2, AMOS22 CT and BTCV. The full specification and the reasons behind each choice are in [PLAN.md](PLAN.md).

The work is a short sequence of steps. Each step is one small script that logs what it did, records the exact inputs it used, and changes nothing outside its output folder. Data preparation runs on a Google Colab runtime; training runs on a single GPU VM.

| Step | Script | What it does | Status |
|---|---|---|---|
| 1 | `cohort_scan.py`, `colab_scan.sh` | Measures every scan and applies the adrenal label cleaning rule | tested locally, not yet run on Colab |
| 2 | (next) | Clean labels, merge to the 9-class map, write the train/test split | not written |
| 3 | (next) | nnU-Net preprocessing to 1.0 mm, publish the cache to Hugging Face | not written |
| 4 | `train.py`, `trainers/adrenal_multiorgan.py` | Training on the GPU VM: one command, continues by itself | checked on CPU, not yet run on a GPU |
| 5 | (next) | Evaluation on the held-out test set | not written |

`vendor/nnUNet` is nnU-Net 2.8.1 plus two SimpleITK resampling modules; see [vendor/README.md](vendor/README.md).

## Step 1: cohort scan

On a Colab runtime (a TPU runtime is convenient because it has many CPUs; the TPU itself is not used):

```bash
git clone https://github.com/ahigazy1/adrenal-multiorgan && cd adrenal-multiorgan
bash colab_scan.sh
```

This downloads TotalSegmentator v2.0.1 from Zenodo (23.6 GB, checksum verified) and the AMOS22 and BTCV label files from Hugging Face at pinned revisions, then writes to `results/`:

- `cohort_scan.csv`: one row per case with organ volumes (mL), each gland's connected components and decision, and the scan-level outcome (`kept_with_adrenal`, `kept_no_adrenal`, `excluded`).
- `cohort_scan.json`: input checksums, package versions, rule thresholds, outcome counts, and the CSV's checksum.
- `cohort_scan.log` and `colab_scan.log`: everything printed during the run.

To try it locally on a few cases: `python cohort_scan.py --out test-results --limit 40 --ts /path/to/Totalsegmentator_dataset_v201.zip`. The script runs a small self-check of the cleaning rule every time it starts.

## Step 4: training

```bash
python train.py
```

The same command starts, continues after an interruption, or restores the latest checkpoint from Hugging Face onto a new machine. Every 100 epochs the latest and best checkpoints and the logs are uploaded. All settings that differ from stock nnU-Net are listed at the top of `trainers/adrenal_multiorgan.py`. To confirm them without a GPU: `python check_trainer.py path/to/plans.json`.

## Data sources and licences

- TotalSegmentator v2.0.1: https://zenodo.org/records/10047292 (CC BY 4.0)
- AMOS22: https://huggingface.co/datasets/MedOtter/amos22-ct-dataset (CC BY 4.0)
- BTCV: https://huggingface.co/datasets/lingheng123/btcv (originally distributed through Synapse under its own terms)
- AtlasNet weights: https://huggingface.co/AbdomenAtlas/AtlasNet
