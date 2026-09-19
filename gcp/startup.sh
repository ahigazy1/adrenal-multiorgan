#!/bin/bash
# Runs on every boot. Restore a verified Hugging Face preprocessing cache, or build and upload it once.
# Then restore/continue training, evaluate the held-out scans and switch off. Checkpoints and the cache
# survive VM/disk deletion only after their uploads have completed successfully.
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
cd /opt
# Freeze the working checkout after preparation, so an existing experiment cannot silently change.
[ -d adrenal-multiorgan ] || git clone https://github.com/ahigazy1/adrenal-multiorgan
cd adrenal-multiorgan
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
pixi run python train.py
pixi run python predict.py
echo "=== all done $(date -u)"
