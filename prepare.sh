#!/bin/bash
# Steps 1-3: download the four public datasets, scan them, build the cleaned dataset and its split, preprocess.
# Run from the repository folder, inside the environment (on the training machine: pixi run bash prepare.sh):
#
#     bash prepare.sh [DATA_FOLDER]        (default: data; needs about 350 GB of free disk)
#
# The tables and logs of steps 1 and 2 are uploaded to Hugging Face before the long preprocessing starts, so the
# exclusions and the split can be looked at straight away. Safe to rerun: finished downloads are skipped.
set -euo pipefail
DATA=$(mkdir -p "${1:-data}" && cd "${1:-data}" && pwd)
MODEL_REPO=${ADRENAL_HF_REPO:-ahigazy1/adrenal-multiorgan-model}
PREPROCESS_WORKERS=${ADRENAL_PREPROCESS_WORKERS:-24}
CPUS=$(nproc)
export HF_XET_HIGH_PERFORMANCE=1
mkdir -p results
echo "=== prepare start $(date -u) on $CPUS CPUs, commit $(git rev-parse HEAD)"
df -h "$DATA" | tail -1

# The four sources come from Hugging Face, one after the other: downloading them at the same time gets
# "429 Too Many Requests" from its CDN. Each is tried up to three times, waiting longer each time.
fetch() { for attempt in 1 2 3; do hf download "$@" && return 0; echo "download failed (attempt $attempt): $1"; sleep $((attempt * 60)); done; return 1; }
# TotalSegmentator v2.0.1 (CC BY 4.0, https://zenodo.org/records/10047292): our copy of the Zenodo file, checked
# against the checksum Zenodo publishes.
TS="$DATA/sources/Totalsegmentator_dataset_v201.zip"
fetch ahigazy1/adrenal-multiorgan-cache sources/Totalsegmentator_dataset_v201.zip --repo-type dataset --local-dir "$DATA"
echo "fe250e5718e0a3b5df4c4ea9d58a62fe  $TS" | md5sum --check
# AMOS22 CT, BTCV and FLARE22, at pinned revisions
fetch MedOtter/amos22-ct-dataset --repo-type dataset --revision c67f7c01e66277038d87975b03b73ece489a3035     --include "train/*" --include "valid/*" --local-dir "$DATA/amos"
fetch lingheng123/btcv --repo-type dataset --revision c1728b451a00c054875a0a97d7658d1eaf8362b5     --include "RawData/Training/*" --local-dir "$DATA/btcv"
fetch MedOtter/FLARE22 --repo-type dataset --revision ab0b99b53e2183fe59321b5888867c6c7cb0792a     --include "images/*" --include "labels/*" --local-dir "$DATA/flare"

echo "=== scan $(date -u)"
python cohort_scan.py --out results --ts "$TS" --amos "$DATA/amos" --btcv "$DATA/btcv" --flare "$DATA/flare" --workers "$CPUS"
echo "=== build dataset $(date -u)"
python build_dataset.py --scan results/cohort_scan.csv --out "$DATA" --ts "$TS" --amos "$DATA/amos" --btcv "$DATA/btcv" --flare "$DATA/flare" --workers "$CPUS"
cp "$DATA"/nnUNet_raw/Dataset902_AdrenalMultiorgan/build_dataset.{csv,json,log} "$DATA"/nnUNet_preprocessed/Dataset902_AdrenalMultiorgan/splits_final.json results/
hf upload "$MODEL_REPO" results dataset --private --commit-message "Cohort scan, dataset and split tables"
echo "=== tables uploaded: https://huggingface.co/$MODEL_REPO/tree/main/dataset"

PREPROCESS_CONSOLE="$DATA/nnUNet_preprocessed/Dataset902_AdrenalMultiorgan/preprocess-console.log"
echo "=== preprocess $(date -u) with $PREPROCESS_WORKERS workers (full console: $PREPROCESS_CONSOLE)"
# nnU-Net's tqdm bar writes carriage returns rather than newlines. If that stream is sent through Google's
# startup-script logger it eventually exceeds the guest agent's scanner token limit and the logger closes the pipe,
# killing preprocessing with SIGPIPE. Keep the raw progress stream in a file and only print sanitized output on error.
if ! python preprocess.py --data "$DATA" --workers "$PREPROCESS_WORKERS" >>"$PREPROCESS_CONSOLE" 2>&1; then
    echo "ERROR: preprocessing failed; last 80 sanitized lines follow"
    tr '\r' '\n' < "$PREPROCESS_CONSOLE" | tail -n 80
    exit 1
fi
echo "=== preprocess done $(date -u)"
echo "=== prepare done $(date -u)"
