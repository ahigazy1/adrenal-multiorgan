"""TotalSegmentator 2.18 vs VISTA3D pseudo-labels, both against each source's own ground truth (380 scans)."""
import csv, json, re, sys
from multiprocessing import Pool
from pathlib import Path
import nibabel as nib, numpy as np
sys.path.insert(0, '/opt/pseudo-run/adrenal-multiorgan/pseudolabels'); sys.path.insert(0, '/opt/pseudo-run/adrenal-multiorgan')
from pseudolabel import D998
TS = Path('/opt/pseudo/pseudolabels/totalseg-2.18.0/bfece4a56f11d1f1'); VI = Path('/opt/vista/out'); DATA = Path('/opt/adrenal-multiorgan/data')
vista = json.load(open('/opt/vista/bundle/docs/labels.json'))
key = lambda n: ' '.join(sorted(re.sub(r'[_\s]+', ' ', n.lower()).replace('vertebrae', 'vertebra').split()))
by_key = {key(n): i for n, i in vista.items()}
# VISTA3D id -> D997/D998 id; kidney cysts merge into kidneys as in the TotalSegmentator recipe
lut = np.zeros(256, np.uint8)
for d, name in enumerate(D998, 1):
    lut[by_key['bladder' if name == 'urinary_bladder' else key(name)]] = d
lut[vista['left kidney cyst']], lut[vista['right kidney cyst']] = D998.index('kidney_left') + 1, D998.index('kidney_right') + 1
GT = {'amos': {1: 'spleen', 2: 'kidney_right', 3: 'kidney_left', 4: 'gallbladder', 5: 'esophagus', 6: 'liver', 7: 'stomach', 8: 'aorta', 9: 'inferior_vena_cava', 10: 'pancreas', 11: 'adrenal_gland_right', 12: 'adrenal_gland_left', 13: 'duodenum', 14: 'urinary_bladder'},
      'btcv': {1: 'spleen', 2: 'kidney_right', 3: 'kidney_left', 4: 'gallbladder', 5: 'esophagus', 6: 'liver', 7: 'stomach', 8: 'aorta', 9: 'inferior_vena_cava', 10: 'portal_vein_and_splenic_vein', 11: 'pancreas', 12: 'adrenal_gland_right', 13: 'adrenal_gland_left'},
      'flare': {1: 'liver', 2: 'kidney_right', 3: 'spleen', 4: 'pancreas', 5: 'aorta', 6: 'inferior_vena_cava', 7: 'adrenal_gland_right', 8: 'adrenal_gland_left', 9: 'gallbladder', 10: 'esophagus', 11: 'stomach', 12: 'duodenum', 13: 'kidney_left'}}
def gt_path(source, case):
    if source == 'amos':
        return next(p for p in (DATA / 'amos/train/labelsTr' / f'{case}.nii.gz', DATA / 'amos/valid/labelsVa' / f'{case}.nii.gz') if p.exists())
    return DATA / ('btcv/RawData/Training/label' if source == 'btcv' else 'flare/labels') / f'{case}.nii.gz'
def score(job):
    source, case = job
    g_img = nib.load(gt_path(source, case)); g = np.asanyarray(g_img.dataobj)
    vname = f'btcv_{case}' if source == 'btcv' else case
    vpath = next(VI.glob(f'*/{vname}/{vname}_trans.nii.gz'))
    preds = {'totalseg': nib.load(TS / source / f'{case}.nii.gz'), 'vista3d': nib.load(vpath)}
    rows = []
    for teacher, img in preds.items():
        assert img.shape == g.shape and np.allclose(img.affine, g_img.affine, atol=1e-3), (teacher, case)
        p = np.asanyarray(img.dataobj)
        if teacher == 'vista3d': p = lut[p]
        for gid, organ in GT[source].items():
            x, y = g == gid, p == D998.index(organ) + 1
            if x.any() or y.any():
                rows.append({'teacher': teacher, 'source': source, 'case': case, 'organ': organ, 'dice': round(2 * (x & y).sum() / (x.sum() + y.sum()), 5),
                             'volume_ratio': round(y.sum() / x.sum(), 4) if x.any() else ''})
    return rows
if __name__ == '__main__':
    jobs = [(s, p.name[:-7]) for s in GT for p in sorted((TS / s).glob('*.nii.gz'))]
    with Pool(24) as pool:
        rows = [r for part in pool.imap_unordered(score, jobs) for r in part]
    out = Path('/opt/pseudo/evaluation/teachers_vs_ground_truth.csv')
    with out.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    from collections import defaultdict
    agg = defaultdict(list)
    for r in rows: agg[(r['organ'], r['source'], r['teacher'])].append(r['dice'])
    organs = sorted({r['organ'] for r in rows})
    print(f"{'organ':30s} " + '  '.join(f'{s:>15s}' for s in GT) + '   (TotalSeg / VISTA3D mean Dice)')
    for o in organs:
        cells = [f"{np.mean(agg[(o, s, 'totalseg')]):.3f}/{np.mean(agg[(o, s, 'vista3d')]):.3f}" if agg[(o, s, 'totalseg')] else '       -       ' for s in GT]
        print(f'{o:30s} ' + '  '.join(f'{c:>15s}' for c in cells))
