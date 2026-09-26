#!/bin/bash
# Download one AbdomenAtlas 3.0 image archive fast (xet), extract only the matched scans, compare voxels.
exec > >(tee -a /var/log/atlas-confirm.log) 2>&1
set -euo pipefail
export HF_XET_HIGH_PERFORMANCE=1 HF_HUB_DISABLE_PROGRESS_BARS=1
D=/opt/atlas-check; mkdir -p $D
T=image_only/AbdomenAtlas3_images_BDMAP_BDMAP_00004409_BDMAP_00004640.tar.gz
echo "=== download $(date -u)"
/opt/pseudo-env/bin/python -c "from huggingface_hub import hf_hub_download as d; d(\"AbdomenAtlas/AbdomenAtlas3.0Mini\", \"$T\", repo_type=\"dataset\", local_dir=\"$D\")"
echo "=== extract $(date -u)"
IDS=$(/opt/pseudo-env/bin/python -c "import json,sys; print(' '.join(json.load(open('/tmp/atlasnet_matched_scans.json'))))")
PATTERNS=$(for i in $IDS; do printf -- "--wildcards *%s* " "$i"; done)
tar -xzf "$D/$T" -C $D $PATTERNS
echo "=== compare $(date -u)"
/opt/pseudo-env/bin/python /tmp/atlasnet_confirmation_compare.py
echo "=== done $(date -u)"; rm -f "$D/$T"
