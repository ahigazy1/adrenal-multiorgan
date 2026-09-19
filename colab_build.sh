#!/bin/bash
# Step 2 on a Colab runtime, after colab_scan.sh: download the AMOS and BTCV images, then run build_dataset.py.
# Run from the repository folder:  bash colab_build.sh
# Needs about 80 GB of free disk. Safe to rerun: finished downloads are skipped, the dataset is rewritten.
set -euo pipefail
DATA=/content/data
test -f results/cohort_scan.csv || { echo "results/cohort_scan.csv is missing: run colab_scan.sh first"; exit 1; }
exec > >(tee -a results/colab_build.log) 2>&1
echo "=== colab_build.sh start $(date -u) on $(nproc) CPUs, commit $(git rev-parse HEAD)"
df -h /content | tail -1

hf download MedOtter/amos22-ct-dataset --repo-type dataset --revision c67f7c01e66277038d87975b03b73ece489a3035 \
    --include "train/imagesTr/*" --include "valid/imagesVa/*" --local-dir "$DATA/amos"
hf download lingheng123/btcv --repo-type dataset --revision c1728b451a00c054875a0a97d7658d1eaf8362b5 \
    --include "RawData/Training/img/*" --local-dir "$DATA/btcv"

python build_dataset.py --scan results/cohort_scan.csv --out "$DATA" --workers "$(nproc)" \
    --ts "$DATA/Totalsegmentator_dataset_v201.zip" --amos "$DATA/amos" --btcv "$DATA/btcv"
cp "$DATA"/nnUNet_raw/Dataset902_AdrenalMultiorgan/build_dataset.{csv,json,log} results/
echo "=== colab_build.sh done $(date -u)"
