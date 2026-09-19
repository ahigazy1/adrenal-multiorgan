#!/bin/bash
# Step 1 on a Colab runtime: download the three public datasets, verify them, run cohort_scan.py.
# Run from the repository folder:  bash colab_scan.sh
# Everything lands in /content/data (inputs) and ./results (outputs + logs). Safe to rerun: finished downloads are skipped.
set -euo pipefail
DATA=/content/data
mkdir -p "$DATA" results
exec > >(tee -a results/colab_scan.log) 2>&1
echo "=== colab_scan.sh start $(date -u) on $(nproc) CPUs, commit $(git rev-parse HEAD)"

pip install --quiet -r requirements.txt

# TotalSegmentator v2.0.1, CC BY 4.0, https://zenodo.org/records/10047292
TS="$DATA/Totalsegmentator_dataset_v201.zip"
if ! echo "fe250e5718e0a3b5df4c4ea9d58a62fe  $TS" | md5sum --check --status 2>/dev/null; then
    wget --continue --no-verbose -O "$TS" "https://zenodo.org/records/10047292/files/Totalsegmentator_dataset_v201.zip?download=1"
    echo "fe250e5718e0a3b5df4c4ea9d58a62fe  $TS" | md5sum --check
fi

# AMOS22 CT and BTCV, pinned to the exact Hugging Face revisions used in the pilot. Labels only: the scan never reads images.
hf download MedOtter/amos22-ct-dataset --repo-type dataset --revision c67f7c01e66277038d87975b03b73ece489a3035 \
    --include "train/labelsTr/*" --include "valid/labelsVa/*" --local-dir "$DATA/amos"
hf download lingheng123/btcv --repo-type dataset --revision c1728b451a00c054875a0a97d7658d1eaf8362b5 \
    --include "RawData/Training/label/*" --local-dir "$DATA/btcv"

python cohort_scan.py --out results --ts "$TS" --amos "$DATA/amos" --btcv "$DATA/btcv" --workers "$(nproc)"
echo "=== colab_scan.sh done $(date -u)"
