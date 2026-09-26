#!/bin/bash
# VISTA3D (MONAI bundle MONAI/vista3d @ c6dbe159) automatic 117-class segmentation of the 380 labelled
# AMOS/BTCV/FLARE22 CTs, 4 GPU processes on separate shards. Raw VISTA3D label ids; mapping to D997 happens later.
exec > >(tee -a /var/log/vista3d.log) 2>&1
set -euo pipefail
echo "=== start $(date -u)"
V=/opt/vista; D=/opt/adrenal-multiorgan/data
mkdir -p $V
[ -x $V/env/bin/python ] || /root/.local/bin/uv venv -q --python 3.12 $V/env
/root/.local/bin/uv pip install -q --python $V/env/bin/python --torch-backend cu128 torch "monai[nibabel]==1.4.0" pytorch-ignite fire einops huggingface_hub
$V/env/bin/python -c "from huggingface_hub import snapshot_download as s; s('MONAI/vista3d', revision='c6dbe159632a4767696e09f91d74d729b82e73e6', local_dir='$V/bundle')"
for k in 0 1 2 3; do mkdir -p $V/in/$k $V/out/$k; done
i=0
for f in $D/amos/train/imagesTr/*.nii.gz $D/amos/valid/imagesVa/*.nii.gz $D/btcv/RawData/Training/img/*.nii.gz $D/flare/images/*.nii.gz; do
    case $f in */amos/*) n=$(basename $f);; */btcv/*) n=btcv_$(basename $f | sed 's/img/label/');; *) n=$(basename $f | sed 's/_0000//');; esac
    ln -sf $f $V/in/$((i % 4))/$n; i=$((i + 1))
done
echo "=== $i scans linked"
cd $V/bundle
for k in 0 1 2 3; do
    $V/env/bin/python -m monai.bundle run --config_file "['configs/inference.json','configs/batch_inference.json']" \
        --bundle_root $V/bundle --input_dir $V/in/$k --output_dir $V/out/$k --output_dtype '$np.uint8' > /var/log/vista3d-$k.log 2>&1 &
done
wait
echo "=== done $(date -u): $(find $V/out -name '*.nii.gz' | wc -l) outputs"
