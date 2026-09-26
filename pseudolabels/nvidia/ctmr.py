"""NV-Segment-CTMR automatic CT_BODY inference on all 793 scans: a one-case gate, then shards in parallel on the GPU.
Same invocation as /opt/ctmr's trial (official bundle, configs/inference.json), batched like /opt/vista/raos-run.py."""
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from cases import cases

ROOT = Path('/opt/nvidia-expand/ctmr')
BUNDLE = Path('/opt/ctmr/upstream/NV-Segment-CTMR')
SHARDS = 6


def prediction(name):
    return ROOT / 'out' / name / f'{name}_trans.nii.gz'


def check(row):
    truth, pred = nib.load(row['label']), nib.load(prediction(row['name']))
    assert pred.shape == truth.shape and np.allclose(pred.affine, truth.affine, atol=1e-3), row['name']
    values = np.asanyarray(pred.dataobj)
    assert values.min() >= 0 and values.max() <= 132 and values.any(), row['name']


def infer(folder):
    command = ['/opt/ctmr/env/bin/python', '-m', 'monai.bundle', 'run',
               '--config_file', "['configs/inference.json','configs/batch_inference.json']",
               '--bundle_root', str(BUNDLE), '--input_dir', str(folder), '--output_dir', str(ROOT / 'out'),
               '--modality', 'CT_BODY', '--output_dtype', '$np.uint8']
    with (ROOT / f'{folder.name}.log').open('a') as log:
        subprocess.run(command, cwd=BUNDLE, stdout=log, stderr=subprocess.STDOUT, check=True)


def link(folder, rows):
    folder.mkdir(parents=True, exist_ok=True)
    for old in folder.glob('*'):
        old.unlink()
    for row in rows:
        (folder / f"{row['name']}.nii.gz").symlink_to(row['image'])


def main():
    rows = cases()
    pending = [r for r in rows if not prediction(r['name']).exists()]
    print(f'{len(rows) - len(pending)}/793 already done', flush=True)
    if pending:
        smoke = pending.pop(0)
        link(ROOT / 'smoke', [smoke])
        infer(ROOT / 'smoke')
        check(smoke)
        print('gate passed:', smoke['name'], flush=True)
    folders = [ROOT / f'in-{k}' for k in range(SHARDS)]
    for k, folder in enumerate(folders):
        link(folder, pending[k::SHARDS])
    with ThreadPoolExecutor(SHARDS) as pool:
        list(pool.map(infer, [f for f in folders if any(f.iterdir())]))
    for row in rows:
        check(row)
    print('COMPLETE: 793 CTMR masks on native grids', flush=True)


if __name__ == '__main__':
    main()
