"""NV-Segment-CT class-plus-one-positive-point adrenal inference on all 793 scans, as /opt/nvct/trial.py did for 3.
Reference-assisted: the point is the reference gland's deepest voxel. Automatic mode is not rerun: the NV-Segment-CT
weights are byte-identical to /opt/vista's VISTA3D (model.pt sha256 c92bab26...), whose masks already exist."""
import csv
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import distance_transform_edt

sys.path.insert(0, str(Path(__file__).parent))
from cases import cases

ROOT = Path('/opt/nvidia-expand/nvct')
NV = Path('/opt/nvct')
WORKERS = 6
# reference ID for (left, right) adrenal per source; VISTA IDs are 9 (left) and 8 (right)
GLANDS = {'amos': (12, 11), 'btcv': (13, 12), 'flare': (8, 7), 'raos': (13, 12)}


def run(job):
    row, organ, gid, pid = job
    out = ROOT / row['name'] / organ
    result = out / 'result.json'
    ref = nib.load(row['label'])
    mask = np.asanyarray(ref.dataobj) == gid
    if not mask.any():
        return None
    if not result.exists():
        out.mkdir(parents=True, exist_ok=True)
        depth = distance_transform_edt(mask, sampling=nib.affines.voxel_sizes(ref.affine))
        point = [int(v) for v in np.unravel_index(depth.argmax(), mask.shape)]
        (out / 'prompt.json').write_text(json.dumps({'points': [point], 'point_labels': [1],
                                                      'source': 'reference_mask_distance_max', 'use_point_window': False}))
        cmd = [str(NV / 'env/bin/python'), str(NV / 'skills/nv-segment-ct/scripts/run_vista3d.py'), row['image'],
               '--output-dir', str(out), '--label-prompts', str(pid), '--points-json', str(out / 'prompt.json')]
        with (out / 'partial.json').open('w') as f, (out / 'inference.log').open('w') as log:
            subprocess.run(cmd, stdout=f, stderr=log, check=True)
        (out / 'partial.json').rename(result)
    summary = json.loads(result.read_text())['output']
    assert summary['label_set_valid'] and all(summary['geometry'][k] for k in ('shape_match', 'spacing_match', 'affine_match'))
    [seg] = out.glob('*/*_seg.nii.gz')
    pred = np.asanyarray(nib.load(seg).dataobj) > 0
    voxel = abs(np.linalg.det(ref.affine[:3, :3])) / 1000
    a, b = int(mask.sum()), int(pred.sum())
    return dict(source=row['source'], case=row['case'], organ=organ, dice=2 * int((mask & pred).sum()) / (a + b),
                reference_ml=a * voxel, predicted_ml=b * voxel, volume_error_ml=(b - a) * voxel,
                volume_error_pct=100 * (b - a) / a)


def main():
    jobs = []
    for row in cases():
        left, right = GLANDS['raos' if row['source'].startswith('raos') else row['source']]
        jobs += [(row, 'adrenal_left', left, 9), (row, 'adrenal_right', right, 8)]
    done = 0
    rows = []
    with ThreadPoolExecutor(WORKERS) as pool:
        for r in pool.map(run, jobs):
            done += 1
            if r:
                rows.append(r)
            if done % 50 == 0:
                print(f'{done}/{len(jobs)} glands', flush=True)
    with (ROOT / 'point_vs_reference.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f'COMPLETE: {len(rows)} glands with a reference', flush=True)


if __name__ == '__main__':
    main()
