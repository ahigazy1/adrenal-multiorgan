#!/bin/bash
# Runs on every boot of the pseudo-label machine: install TotalSegmentator, label every scan not yet on Hugging Face,
# upload, switch off. Set instance metadata hold=1 to boot without running anything or switching off.
exec > >(tee -a /var/log/pseudo.log | tr '\r' '\n') 2>&1
echo "=== start $(date -u)"
metadata() { curl -sf -H "Metadata-Flavor: Google" "http://metadata.google.internal/computeMetadata/v1/$1"; }
if metadata instance/attributes/hold >/dev/null; then echo "hold is set: doing nothing"; exit 0; fi
trap 'echo "=== switching off $(date -u)"; shutdown -h now' EXIT
set -euo pipefail
export HOME=/root PATH=/root/.local/bin:$PATH PYTHONUNBUFFERED=1 HF_HUB_DISABLE_PROGRESS_BARS=1 HF_XET_HIGH_PERFORMANCE=1

cd /opt
[ -d adrenal-multiorgan ] || git clone https://github.com/ahigazy1/adrenal-multiorgan
git -C adrenal-multiorgan pull --ff-only origin main
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -x /opt/pseudo-env/bin/python ] || uv venv --python 3.12 /opt/pseudo-env
uv pip install --python /opt/pseudo-env/bin/python TotalSegmentator==2.18.0 "huggingface_hub[hf_xet]>=1.0"
source /opt/pseudo-env/bin/activate

ACCESS=$(metadata instance/service-accounts/default/token | python3 -c "import json,sys; print(json.load(sys.stdin)['access_token'])")
PROJECT=$(metadata project/project-id)
HF_TOKEN=$(curl -sf -H "Authorization: Bearer $ACCESS" \
    "https://secretmanager.googleapis.com/v1/projects/$PROJECT/secrets/HF_TOKEN/versions/latest:access" \
    | python3 -c "import base64,json,sys; print(base64.b64decode(json.load(sys.stdin)['payload']['data']).decode().strip())")
[ -n "$HF_TOKEN" ] || { echo "ERROR: could not read the Hugging Face token from Secret Manager"; exit 1; }
export HF_TOKEN

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
totalseg_download_weights -t total   # once, before the workers start, so they do not all download at the same time
python adrenal-multiorgan/pseudolabels/pseudolabel.py --data /opt/pseudo --workers "${PSEUDO_WORKERS:-4}"
echo "=== all done $(date -u)"
