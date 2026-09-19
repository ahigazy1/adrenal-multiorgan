#!/bin/bash
# Runs on the training machine at every start (create_vm.sh installs it as the machine's startup script).
# First start: install the pinned environment, then download, clean and preprocess the data (prepare.sh, a few
# hours). Every start: train or continue, and when training has finished, segment the held-out test scans. Then the machine switches itself off, so it never
# sits idle on the bill. Whatever goes wrong, it also switches off; nothing restarts it automatically.
#
# Everything printed goes to /var/log/adrenal.log and to the machine's serial console, which Google keeps in
# Cloud Logging, so progress.sh can show the reason for a failure even after the machine is off.
#
# To keep the machine on for a look around (no training, no switching off), set the metadata key "hold":
#     gcloud compute instances add-metadata adrenal-train --zone=ZONE --metadata=hold=1     (remove-metadata --keys=hold to undo)
exec > >(tee -a /var/log/adrenal.log | tr '\r' '\n') 2>&1
echo "=== start $(date -u)"
metadata() { curl -sf -H "Metadata-Flavor: Google" "http://metadata.google.internal/computeMetadata/v1/$1"; }
if metadata instance/attributes/hold >/dev/null; then echo "hold is set: doing nothing"; exit 0; fi
trap 'echo "=== switching off $(date -u)"; shutdown -h now' EXIT
set -euo pipefail

export HOME=/root PATH=/root/.pixi/bin:$PATH
export PIXI_LOCKED=true   # pixi must use pixi.lock exactly, never re-solve the environment
export PYTHONUNBUFFERED=1 nnUNet_n_proc_DA=32 TORCHINDUCTOR_COMPILE_THREADS=16
cd /opt
# The code is frozen once preparation is complete because train.py refuses to continue a run whose code changed.
# While preparation is incomplete, pull main so preprocessing/restart fixes are picked up after an interruption.
[ -d adrenal-multiorgan ] || git clone https://github.com/ahigazy1/adrenal-multiorgan
cd adrenal-multiorgan
if [ ! -f data/.prepared ]; then
    git pull --ff-only origin main
fi
command -v pixi >/dev/null || curl -fsSL https://pixi.sh/install.sh | PIXI_NO_PATH_UPDATE=1 bash
pixi install

# The Hugging Face token lives in Secret Manager; it is held in memory only and never printed.
ACCESS=$(metadata instance/service-accounts/default/token | python3 -c "import json,sys; print(json.load(sys.stdin)['access_token'])")
PROJECT=$(metadata project/project-id)
HF_TOKEN=$(curl -sf -H "Authorization: Bearer $ACCESS" \
    "https://secretmanager.googleapis.com/v1/projects/$PROJECT/secrets/HF_TOKEN/versions/latest:access" \
    | python3 -c "import base64,json,sys; print(base64.b64decode(json.load(sys.stdin)['payload']['data']).decode().strip())")
[ -n "$HF_TOKEN" ] || { echo "ERROR: could not read the Hugging Face token from Secret Manager"; exit 1; }
export HF_TOKEN

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
if [ ! -f data/.prepared ]; then   # after an interruption: downloads continue, the later steps start again
    pixi run bash prepare.sh data
    touch data/.prepared
fi
pixi run python train.py
pixi run python predict.py
echo "=== all done $(date -u)"
