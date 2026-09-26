"""The 380 AMOS/BTCV/FLARE scans the TotalSegmentator and VISTA3D teachers ran on, plus the 413 RAOS scans.
Output names follow /opt/vista: btcv cases get a btcv_ prefix."""
import json
from pathlib import Path

DATA = Path('/opt/adrenal-multiorgan/data')
TS = Path('/opt/pseudo/pseudolabels/totalseg-2.18.0/bfece4a56f11d1f1')


def paths(source, case):
    if source == 'amos':
        label = next(p for p in (DATA / 'amos/train/labelsTr' / f'{case}.nii.gz', DATA / 'amos/valid/labelsVa' / f'{case}.nii.gz') if p.exists())
        return Path(str(label).replace('labelsTr', 'imagesTr').replace('labelsVa', 'imagesVa')), label
    if source == 'btcv':
        return (DATA / 'btcv/RawData/Training/img' / f"{case.replace('label', 'img')}.nii.gz",
                DATA / 'btcv/RawData/Training/label' / f'{case}.nii.gz')
    return DATA / 'flare/images' / f'{case}_0000.nii.gz', DATA / 'flare/labels' / f'{case}.nii.gz'


def cases():
    rows = []
    for source in ('amos', 'btcv', 'flare'):
        for p in sorted((TS / source).glob('*.nii.gz')):
            case = p.name[:-7]
            image, label = paths(source, case)
            rows.append(dict(source=source, case=case, name=f'btcv_{case}' if source == 'btcv' else case,
                             image=str(image), label=str(label)))
    for r in json.loads(Path('/opt/vista/raos/cases.json').read_text()):
        rows.append(dict(source=r['source'], case=r['case'], name=r['case'], image=r['image'], label=r['label']))
    assert len(rows) == 793 and len({r['name'] for r in rows}) == 793
    assert all(Path(r['image']).exists() and Path(r['label']).exists() for r in rows)
    return rows
