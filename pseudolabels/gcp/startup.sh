#!/bin/bash
# Runs on every boot of the pseudo-label machine: install TotalSegmentator, label every scan not yet on Hugging Face,
# upload, switch off. Set instance metadata hold=1 to boot without running anything or switching off.
exec > >(tee -a /var/log/pseudo.log | tr '\r' '\n') 2>&1
echo "=== start $(date -u)"
metadata() { curl -sf -H "Metadata-Flavor: Google" "http://metadata.google.internal/computeMetadata/v1/$1"; }
if metadata instance/attributes/hold >/dev/null; then echo "hold is set: doing nothing"; exit 0; fi
trap 'status=$?; echo "=== exit=$status; switching off $(date -u)"; shutdown -h now' EXIT
set -euo pipefail
export PATH=/root/.local/bin:$PATH PYTHONUNBUFFERED=1 HF_HUB_DISABLE_PROGRESS_BARS=1 HF_XET_HIGH_PERFORMANCE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export SITK_THREADS=4 ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=4 nnUNet_def_n_proc=4

cd /opt
REVISION=$(metadata instance/attributes/pseudo-revision)
[[ "$REVISION" =~ ^[0-9a-f]{40}$ ]] || { echo "ERROR: missing pinned pseudo-revision"; exit 1; }
[ -d adrenal-multiorgan ] || git clone https://github.com/ahigazy1/adrenal-multiorgan
git -C adrenal-multiorgan fetch origin "$REVISION"
git -C adrenal-multiorgan checkout --detach "$REVISION"
echo "=== code revision: $REVISION"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -x /opt/pseudo-env/bin/python ] || uv venv --python 3.12 /opt/pseudo-env
uv pip sync --python /opt/pseudo-env/bin/python --torch-backend cu128 \
    adrenal-multiorgan/pseudolabels/requirements.txt
source /opt/pseudo-env/bin/activate
# Preserve the existing custom modules for the later dataset/training stage; teacher plans stay stock.
python - <<'PY'
from pathlib import Path
import shutil
import nnunetv2
source = Path('/opt/adrenal-multiorgan/vendor/nnUNet/nnunetv2/preprocessing/resampling')
target = Path(nnunetv2.__file__).parent / 'preprocessing/resampling'
for name in ('cucim_resampling.py', 'sitk_resampling.py', 'sitk_logits_resampling.py'):
    shutil.copy2(source / name, target / name)
import torch
assert torch.cuda.is_available(), 'CUDA unavailable: refusing CPU fallback'
# Exercise an actual CUDA convolution, not just driver detection.
torch.nn.Conv3d(1, 2, 3).cuda()(torch.zeros(1, 1, 8, 8, 8, device='cuda'))
torch.cuda.synchronize()
print('=== CUDA convolution passed', torch.cuda.get_device_name(0), flush=True)
PY

ACCESS=$(metadata instance/service-accounts/default/token | python3 -c "import json,sys; print(json.load(sys.stdin)['access_token'])")
PROJECT=$(metadata project/project-id)
HF_TOKEN=$(curl -sf -H "Authorization: Bearer $ACCESS" \
    "https://secretmanager.googleapis.com/v1/projects/$PROJECT/secrets/HF_TOKEN/versions/latest:access" \
    | python3 -c "import base64,json,sys; print(base64.b64decode(json.load(sys.stdin)['payload']['data']).decode().strip())")
[ -n "$HF_TOKEN" ] || { echo "ERROR: could not read the Hugging Face token from Secret Manager"; exit 1; }
export HF_TOKEN

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
totalseg_download_weights -t total   # once, before the workers start, so they do not all download at the same time
python adrenal-multiorgan/pseudolabels/check_pseudolabel.py
WORKERS=$(metadata instance/attributes/pseudo-workers || echo 24)
python adrenal-multiorgan/pseudolabels/pseudolabel.py --data /opt/pseudo --workers "$WORKERS"
echo "=== all done $(date -u)"
