#!/bin/bash
# Startup script for the D997 fine-tune on the existing adrenal-train VM: build Dataset903, preprocess with D997's
# plans, train (ADRENAL_RUN=d997), switch off. Runs on every boot; each finished step is skipped, and training resumes
# from its last checkpoint. The code is the commit in instance metadata finetune-revision (never a moving branch).
# Uses the training disk's data (/opt/adrenal-multiorgan/data) and the pseudo-labels in /opt/pseudo; the frozen
# training checkout in /opt/adrenal-multiorgan is not touched. Set instance metadata hold=1 to boot without running.
exec > >(tee -a /var/log/finetune.log | tr '\r' '\n') 2>&1
echo "=== start $(date -u)"
metadata() { curl -sf -H "Metadata-Flavor: Google" "http://metadata.google.internal/computeMetadata/v1/$1"; }
if metadata instance/attributes/hold >/dev/null; then echo "hold is set: doing nothing"; exit 0; fi
trap 'status=$?; echo "=== exit=$status; switching off $(date -u)"; shutdown -h now' EXIT
set -euo pipefail
export HOME=/root PATH=/root/.pixi/bin:$PATH PIXI_LOCKED=true PYTHONUNBUFFERED=1 HF_HUB_DISABLE_PROGRESS_BARS=1
export ADRENAL_RUN=d997 ADRENAL_DATA=/opt/adrenal-multiorgan/data
PSEUDO=/opt/pseudo/pseudolabels/totalseg-2.18.0/bfece4a56f11d1f1

REVISION=$(metadata instance/attributes/finetune-revision)
[[ "$REVISION" =~ ^[0-9a-f]{40}$ ]] || { echo "ERROR: missing pinned finetune-revision"; exit 1; }
cd /opt
[ -d finetune-run ] || git clone -q https://github.com/ahigazy1/adrenal-multiorgan finetune-run
git -C finetune-run fetch -q origin "$REVISION"
git -C finetune-run checkout -q --detach "$REVISION"
cd finetune-run
echo "=== code revision: $REVISION"
pixi install

ACCESS=$(metadata instance/service-accounts/default/token | python3 -c "import json,sys; print(json.load(sys.stdin)['access_token'])")
HF_TOKEN=$(curl -sf -H "Authorization: Bearer $ACCESS" \
    "https://secretmanager.googleapis.com/v1/projects/$(metadata project/project-id)/secrets/HF_TOKEN/versions/latest:access" \
    | python3 -c "import base64,json,sys; print(base64.b64decode(json.load(sys.stdin)['payload']['data']).decode().strip())")
[ -n "$HF_TOKEN" ] || { echo "ERROR: could not read the Hugging Face token from Secret Manager"; exit 1; }
export HF_TOKEN
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

RAW=$ADRENAL_DATA/nnUNet_raw/Dataset903_D997Finetune
[ -f "$RAW/build_dataset.json" ] || pixi run python finetune.py build --data "$ADRENAL_DATA" \
    --scan /opt/adrenal-multiorgan/results/cohort_scan.csv --pseudo "$PSEUDO"
# nnU-Net's progress bars write carriage returns; keep them in a file rather than the startup-script logger.
[ -f "$ADRENAL_DATA/nnUNet_preprocessed/Dataset903_D997Finetune/.done" ] || \
    pixi run python finetune.py preprocess --data "$ADRENAL_DATA" >> /var/log/finetune-preprocess.log 2>&1
pixi run python train.py
echo "=== all done $(date -u)"
