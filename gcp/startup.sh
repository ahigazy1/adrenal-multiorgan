#!/bin/bash
# Runs on every boot. Restore a verified copy of the prepared data from Hugging Face, or prepare it here
# (prepare.sh). Then train or continue training while the prepared data uploads in the background, evaluate the
# held-out scans and switch off. Checkpoints and the prepared data survive deleting the VM and its disk only
# once their uploads have completed.
# Raw console output is kept locally; carriage returns are normalized before the guest-agent logger.
# Set instance metadata hold=1 to boot without running anything or switching off.
exec > >(tee -a /var/log/adrenal.log | tr '\r' '\n') 2>&1
echo "=== start $(date -u)"
metadata() { curl -sf -H "Metadata-Flavor: Google" "http://metadata.google.internal/computeMetadata/v1/$1"; }
if metadata instance/attributes/hold >/dev/null; then echo "hold is set: doing nothing"; exit 0; fi
trap 'echo "=== switching off $(date -u)"; shutdown -h now' EXIT
set -euo pipefail

export HOME=/root PATH=/root/.pixi/bin:$PATH
export PIXI_LOCKED=true
export PYTHONUNBUFFERED=1 nnUNet_n_proc_DA=32 TORCHINDUCTOR_COMPILE_THREADS=16
export HF_HUB_DISABLE_PROGRESS_BARS=1
export NUMEXPR_MAX_THREADS=48   # let numexpr use every core instead of its default cap of 16
cd /opt
# Freeze the working checkout after preparation, so an existing experiment cannot silently change.
[ -d adrenal-multiorgan ] || git clone https://github.com/ahigazy1/adrenal-multiorgan
cd adrenal-multiorgan
# Prepared data without nnU-Net's dataset fingerprint is incomplete (training stops on it): prepare again.
# Downloads and finished scans are kept, so that takes minutes, and the newest code is pulled for it.
[ -f data/nnUNet_preprocessed/Dataset902_AdrenalMultiorgan/dataset_fingerprint.json ] || rm -f data/.prepared data/.preprocessed-local.json
if [ ! -f data/.prepared ]; then
    git pull --ff-only origin main
fi
command -v pixi >/dev/null || curl -fsSL https://pixi.sh/install.sh | PIXI_NO_PATH_UPDATE=1 bash
pixi install

# HF_TOKEN must have read/write access to BOTH the private cache dataset and the model repository.
# The token comes from Secret Manager and is never written to the repository or printed.
ACCESS=$(metadata instance/service-accounts/default/token | python3 -c "import json,sys; print(json.load(sys.stdin)['access_token'])")
PROJECT=$(metadata project/project-id)
HF_TOKEN=$(curl -sf -H "Authorization: Bearer $ACCESS" \
    "https://secretmanager.googleapis.com/v1/projects/$PROJECT/secrets/HF_TOKEN/versions/latest:access" \
    | python3 -c "import base64,json,sys; print(base64.b64decode(json.load(sys.stdin)['payload']['data']).decode().strip())")
[ -n "$HF_TOKEN" ] || { echo "ERROR: could not read the Hugging Face token from Secret Manager"; exit 1; }
export HF_TOKEN

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
# This validates local completion, or restores a complete compatible cache, or builds and publishes one.
# A plain touch is not enough: .prepared is written only after successful verification and publication.
pixi run python hf_cache.py ensure --data data
# The upload of the prepared data (about 170 GB) is only for a future replacement machine, so it runs beside
# training at the lowest CPU and disk priority instead of making the GPU wait. It resumes on every start.
nice -n 19 ionice -c 3 pixi run python hf_cache.py publish --data data >>/var/log/adrenal-cache-upload.log 2>&1 &
UPLOAD=$!
pixi run python train.py
pixi run python predict.py
echo "=== waiting for the cache upload, if it is still running (see /var/log/adrenal-cache-upload.log)"
wait "$UPLOAD" || echo "WARNING: the cache upload failed; training results are not affected"
echo "=== all done $(date -u)"
