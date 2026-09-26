#!/bin/bash
# RAOS images + labels (ahigazy1/AdrenalSeg-Sources RAOS-Real, the copy compare.py uses) onto the training disk.
exec > >(tee -a /var/log/raos-fetch.log) 2>&1
set -euo pipefail
echo "=== start $(date -u)"
metadata() { curl -sf -H "Metadata-Flavor: Google" "http://metadata.google.internal/computeMetadata/v1/$1"; }
ACCESS=$(metadata instance/service-accounts/default/token | python3 -c "import json,sys; print(json.load(sys.stdin)['access_token'])")
export HF_TOKEN=$(curl -sf -H "Authorization: Bearer $ACCESS" "https://secretmanager.googleapis.com/v1/projects/adrenal-seg/secrets/HF_TOKEN/versions/latest:access" | python3 -c "import base64,json,sys; print(base64.b64decode(json.load(sys.stdin)['payload']['data']).decode().strip())")
export HF_XET_HIGH_PERFORMANCE=1 HF_HUB_DISABLE_PROGRESS_BARS=1
/opt/pseudo-env/bin/python - <<'PY'
from huggingface_hub import HfApi, snapshot_download
rev = HfApi().dataset_info('ahigazy1/AdrenalSeg-Sources').sha
print('AdrenalSeg-Sources revision', rev, flush=True)
snapshot_download('ahigazy1/AdrenalSeg-Sources', repo_type='dataset', revision=rev, local_dir='/opt/adrenal-multiorgan/data/raos-source',
                  allow_patterns=['RAOS-Real/*/images*/*', 'RAOS-Real/*/labels*/*', 'RAOS-Real/*.json', 'RAOS-Real/*.md', 'RAOS-Real/*/*.json'])
PY
echo "=== done $(date -u): $(find /opt/adrenal-multiorgan/data/raos-source -name '*.nii.gz' | wc -l) NIfTI files"
