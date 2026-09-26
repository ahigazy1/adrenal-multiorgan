"""Score CTMR like the TotalSegmentator/VISTA3D comparison (/tmp/teachers.py, same lookup table: CTMR shares VISTA3D's
label IDs) and like /opt/vista/raos-run.py on RAOS; then print the Dice and volume tables for all teachers."""
import csv
import sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import nibabel as nib
import numpy as np

sys.path.insert(0, '/tmp'); sys.path.insert(0, str(Path(__file__).parent))
import teachers as T
from cases import cases

ROOT = Path('/opt/nvidia-expand')
RAOS = {'liver': (1, 1), 'spleen': (2, 3), 'kidney_left': (3, 14), 'kidney_right': (4, 5), 'pancreas': (8, 4),
        'adrenal_right': (12, 8), 'adrenal_left': (13, 9)}  # as /opt/vista/raos-run.py


def score(row):
    img, ref = nib.load(ROOT / 'ctmr/out' / row['name'] / f"{row['name']}_trans.nii.gz"), nib.load(row['label'])
    assert img.shape == ref.shape and np.allclose(img.affine, ref.affine, atol=1e-3)
    p, g = np.asanyarray(img.dataobj), np.asanyarray(ref.dataobj)
    voxel = abs(np.linalg.det(ref.affine[:3, :3])) / 1000
    out = []
    if row['source'].startswith('raos'):
        for organ, (gid, pid) in RAOS.items():
            x, y = g == gid, p == pid
            y |= p == 116 if organ == 'kidney_left' else (p == 117 if organ == 'kidney_right' else False)
            a, b = int(x.sum()), int(y.sum())
            out.append(dict(case=row['case'], source=row['source'], organ=organ, dice=2 * int((x & y).sum()) / (a + b) if a + b else '',
                            reference_ml=a * voxel, predicted_ml=b * voxel, volume_error_ml=(b - a) * voxel,
                            volume_error_pct=100 * (b - a) / a if a else ''))
        return out
    p = T.lut[p]
    for gid, organ in T.GT[row['source']].items():
        x, y = g == gid, p == T.D998.index(organ) + 1
        if x.any() or y.any():
            out.append({'teacher': 'ctmr', 'source': row['source'], 'case': row['case'], 'organ': organ,
                        'dice': round(2 * (x & y).sum() / (x.sum() + y.sum()), 5),
                        'volume_ratio': round(y.sum() / x.sum(), 4) if x.any() else ''})
    return out


def write(path, rows):
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


if __name__ == '__main__':
    with Pool(24) as pool:
        rows = [r for part in pool.map(score, cases()) for r in part]
    main = [r for r in rows if 'teacher' in r]
    write(ROOT / 'ctmr_vs_ground_truth.csv', main)
    write(ROOT / 'ctmr_raos_per_case.csv', [r for r in rows if 'teacher' not in r])
    allrows = list(csv.DictReader(open('/opt/pseudo/evaluation/teachers_vs_ground_truth.csv'))) + main
    agg = defaultdict(list)
    for r in allrows:
        agg[(r['organ'], r['source'], r['teacher'])].append(r)
    print(f"{'organ':28s} {'source':6s} {'n':>4s}   mean Dice TS / VISTA3D / CTMR     median volume ratio TS / VISTA3D / CTMR")
    for organ in sorted({r['organ'] for r in allrows}):
        for source in T.GT:
            if not agg[(organ, source, 'ctmr')]:
                continue
            d = [np.mean([float(r['dice']) for r in agg[(organ, source, t)]]) for t in ('totalseg', 'vista3d', 'ctmr')]
            v = [np.median([float(r['volume_ratio']) for r in agg[(organ, source, t)] if r['volume_ratio']]) for t in ('totalseg', 'vista3d', 'ctmr')]
            print(f"{organ:28s} {source:6s} {len(agg[(organ, source, 'ctmr')]):4d}   {d[0]:.3f} / {d[1]:.3f} / {d[2]:.3f}"
                  f"          {v[0]:.3f} / {v[1]:.3f} / {v[2]:.3f}")
    vista = list(csv.DictReader(open('/opt/vista/raos/per_case.csv')))
    ctmr = [r for r in rows if 'teacher' not in r]
    print('\nRAOS (413): mean Dice VISTA3D / CTMR, median signed volume error % VISTA3D / CTMR')
    for organ in RAOS:
        stats = []
        for data in (vista, ctmr):
            sel = [r for r in data if r['organ'] == organ and r['dice'] != '']
            stats.append((np.mean([float(r['dice']) for r in sel]),
                          np.median([float(r['volume_error_pct']) for r in sel if r['volume_error_pct'] != ''])))
        print(f'{organ:16s} {stats[0][0]:.3f} / {stats[1][0]:.3f}    {stats[0][1]:+.1f} / {stats[1][1]:+.1f}')
