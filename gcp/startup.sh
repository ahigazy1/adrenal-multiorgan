#!/bin/bash
# Runs on the training machine at every start (create_vm.sh installs it as the machine's startup script).
# First start: install the pinned environment and download the cache. Every start: train or continue, and when
# training has finished, segment the held-out test scans. Then the machine switches itself off, so it never
# sits idle on the bill. Everything printed goes to /var/log/adrenal.log.
#
# To keep the machine on for a look around (no training, no switching off), set the metadata key "hold":
#     gcloud compute instances add-metadata adrenal-train --zone=ZONE --metadata=hold=1     (remove-metadata --keys=hold to undo)
exec >>/var/log/adrenal.log 2>&1
echo "=== start $(date -u)"
metadata() { curl -sf -H "Metadata-Flavor: Google" "http://metadata.google.internal/computeMetadata/v1/$1"; }
if metadata instance/attributes/hold >/dev/null; then echo "hold is set: doing nothing"; exit 0; fi
trap 'echo "=== switching off $(date -u)"; shutdown -h now' EXIT
set -euo pipefail

export HOME=/root PATH=/root/.pixi/bin:$PATH
export PYTHONUNBUFFERED=1 nnUNet_n_proc_DA=32 TORCHINDUCTOR_COMPILE_THREADS=16
cd /opt
# The code is cloned once and never updated: train.py refuses to continue a run whose code has changed.
[ -d adrenal-multiorgan ] || git clone https://github.com/ahigazy1/adrenal-multiorgan
cd adrenal-multiorgan
command -v pixi >/dev/null || curl -fsSL https://pixi.sh/install.sh | bash
pixi install --locked

# The Hugging Face token lives in Secret Manager; it is held in memory only.
ACCESS=$(metadata instance/service-accounts/default/token | python3 -c "import json,sys; print(json.load(sys.stdin)['access_token'])")
PROJECT=$(metadata project/project-id)
export HF_TOKEN=$(curl -sf -H "Authorization: Bearer $ACCESS" \
    "https://secretmanager.googleapis.com/v1/projects/$PROJECT/secrets/HF_TOKEN/versions/latest:access" \
    | python3 -c "import base64,json,sys; print(base64.b64decode(json.load(sys.stdin)['payload']['data']).decode().strip())")

if [ ! -f data/.cache-complete ]; then
    pixi run hf download ahigazy1/adrenal-multiorgan-cache --repo-type dataset --local-dir data --exclude "sources/*"
    touch data/.cache-complete
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
pixi run python train.py
pixi run python predict.py
echo "=== all done $(date -u)"
