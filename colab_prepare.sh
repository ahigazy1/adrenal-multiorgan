#!/bin/bash
# Steps 1 and 2 on a Colab runtime: download the four public datasets, verify them, scan, build the dataset.
# Run from the repository folder:  bash colab_prepare.sh
# Inputs land in /content/data, logs and tables in ./results. Needs about 80 GB of free disk.
# Safe to rerun: finished downloads are skipped.
set -euo pipefail
DATA=/content/data
export HF_XET_HIGH_PERFORMANCE=1
mkdir -p "$DATA" results
exec > >(tee -a results/colab_prepare.log) 2>&1
echo "=== colab_prepare.sh start $(date -u) on $(nproc) CPUs, commit $(git rev-parse HEAD)"
df -h /content | tail -1

pip install --quiet -r requirements.txt

# TotalSegmentator v2.0.1, CC BY 4.0, https://zenodo.org/records/10047292. Zenodo is slow, so a copy of the same
# file in our Hugging Face repository is tried first; either way the checksum Zenodo publishes must match.
TS="$DATA/sources/Totalsegmentator_dataset_v201.zip"
if ! echo "fe250e5718e0a3b5df4c4ea9d58a62fe  $TS" | md5sum --check --status 2>/dev/null; then
    hf download ahigazy1/adrenal-multiorgan-cache sources/Totalsegmentator_dataset_v201.zip --repo-type dataset --local-dir "$DATA" \
        || { mkdir -p "$DATA/sources"; wget --continue --no-verbose -O "$TS" "https://zenodo.org/records/10047292/files/Totalsegmentator_dataset_v201.zip?download=1"; }
    echo "fe250e5718e0a3b5df4c4ea9d58a62fe  $TS" | md5sum --check
fi

# AMOS22 CT, BTCV and FLARE22 from Hugging Face, at pinned revisions
hf download MedOtter/amos22-ct-dataset --repo-type dataset --revision c67f7c01e66277038d87975b03b73ece489a3035 \
    --include "train/*" --include "valid/*" --local-dir "$DATA/amos"
hf download lingheng123/btcv --repo-type dataset --revision c1728b451a00c054875a0a97d7658d1eaf8362b5 \
    --include "RawData/Training/*" --local-dir "$DATA/btcv"
hf download MedOtter/FLARE22 --repo-type dataset --revision ab0b99b53e2183fe59321b5888867c6c7cb0792a \
    --include "images/*" --include "labels/*" --local-dir "$DATA/flare"

python cohort_scan.py --out results --ts "$TS" --amos "$DATA/amos" --btcv "$DATA/btcv" --flare "$DATA/flare" --workers "$(nproc)"
python build_dataset.py --scan results/cohort_scan.csv --out "$DATA" --ts "$TS" --amos "$DATA/amos" --btcv "$DATA/btcv" --flare "$DATA/flare" --workers "$(nproc)"
cp "$DATA"/nnUNet_raw/Dataset902_AdrenalMultiorgan/build_dataset.{csv,json,log} results/
echo "=== colab_prepare.sh done $(date -u)"
