"""Score completed AtlasNet TS predictions against raw masks, stratified by Dataset902 membership.

python evaluate_totalseg.py --run /opt/atlasnet-totalseg --data /opt/adrenal-multiorgan/data

Reads individual masks directly from the source ZIP. No speckle removal or overlap precedence:
these scores differ deliberately from evaluation against Dataset902's cleaned labels.
"""
import argparse
import csv
import gzip
import json
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import nibabel as nib
import numpy as np
import polars as pl

from build_dataset import DATASET, LABELS
from cohort_scan import TS_FILES, open_zip, sha256
from evaluate import volume_metrics

FOCUS = ['adrenal_left', 'adrenal_right', 'aorta', 'pancreas', 'liver']


def memberships(build, splits):
    with build.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    fold = json.loads(splits.read_text())[0]
    train, val = set(fold['train']), set(fold['val'])
    kept = {r['name'] for r in rows if r['role'] == 'train'}
    test = {r['name'] for r in rows if r['role'] == 'test'}
    if train & val or (train | val) != kept or test & kept:
        raise ValueError('Dataset902 build and fold-0 membership disagree')
    return {**dict.fromkeys(train, 'train'), **dict.fromkeys(val, 'validation'), **dict.fromkeys(test, 'test')}


def score(record, run, archive, roles):
    case = record['case']
    paths = list(run.glob(f'worker-*/native_labels/{case}.nii.gz'))
    if len(paths) != 1 or sha256(paths[0]) != record['sha256']:
        raise ValueError(f'{case}: missing, duplicate or changed prediction')
    prediction = nib.load(paths[0])
    values = np.asanyarray(prediction.dataobj)
    if not np.isin(values, list(LABELS.values())).all():
        raise ValueError(f'{case}: unexpected prediction labels')
    source_case = case.removeprefix('ts_')
    rows = []
    for organ, structures in TS_FILES.items():
        truth = np.zeros(prediction.shape, dtype=bool)
        for structure in structures:
            member = f'{source_case}/segmentations/{structure}.nii.gz'
            reference = nib.Nifti1Image.from_bytes(gzip.decompress(open_zip(archive).read(member)))
            if reference.shape != prediction.shape or not np.allclose(reference.affine, prediction.affine, atol=1e-3):
                raise ValueError(f'{case}/{organ}: reference/prediction grid mismatch')
            truth |= np.asanyarray(reference.dataobj) > 0
        predicted = values == LABELS[organ]
        voxel_ml = float(abs(np.linalg.det(prediction.affine[:3, :3]))) / 1000
        rows.append({'case': case, 'split': roles.get(case, 'excluded_from_dataset902'), 'organ': organ,
                     **volume_metrics(int(truth.sum()), int(predicted.sum()), int((truth & predicted).sum()), voxel_ml)})
    return rows


def summarise(frame, groups):
    columns = ['dice', 'volume_error_ml', 'absolute_volume_error_ml', 'volume_error_pct', 'absolute_volume_error_pct']
    return frame.group_by(groups).agg(
        pl.len().alias('cases'), pl.col('dice').count().alias('scored'), pl.col('miss').sum(),
        pl.col('volume_error_pct').count().alias('reference_present'),
        *[expr for name in columns for expr in (pl.col(name).mean().alias(f'{name}_mean'),
                                                 pl.col(name).median().alias(f'{name}_median'))]).sort(groups)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    completion = args.run / 'complete.json'
    evidence = json.loads(completion.read_text())  # only a completed inference run is scored
    records = evidence['outputs']
    cases = json.loads((args.run / 'cases.json').read_text())
    if len(records) != 1228 or len({r['case'] for r in records}) != 1228 or {r['case'] for r in records} != set(cases):
        raise ValueError('Incomplete or duplicate inference manifest')
    build = args.data / 'nnUNet_raw' / DATASET / 'build_dataset.csv'
    splits = args.data / 'nnUNet_preprocessed' / DATASET / 'splits_final.json'
    archive = args.data / 'sources/Totalsegmentator_dataset_v201.zip'
    roles = memberships(build, splits)
    if not {case for case in roles if case.startswith('ts_')} <= set(cases):
        raise ValueError('Dataset902 TS cases missing from inference cohort')
    output = args.run / 'evaluation'
    output.mkdir(exist_ok=True)
    (output / 'complete.json').unlink(missing_ok=True)
    with Pool(args.workers) as pool:
        rows = []
        for index, result in enumerate(pool.imap(partial(score, run=args.run, archive=archive, roles=roles), records), 1):
            rows.extend(result)
            if index % 50 == 0:
                print(f'Scored {index}/{len(records)}', flush=True)
    frame = pl.DataFrame(rows, infer_schema_length=None)
    frame.write_csv(output / 'per_case.csv')
    summarise(frame, ['split', 'organ']).write_csv(output / 'by_split.csv')
    summarise(frame.filter(pl.col('organ').is_in(FOCUS)), ['split', 'organ']).write_csv(output / 'focus_by_split.csv')
    summarise(frame, ['organ']).write_csv(output / 'all_scans_descriptive.csv')
    record = {'cases': len(records), 'rows': len(rows), 'checkpoint_sha256': evidence['run']['checkpoint_sha256'],
              'reference': 'raw TotalSegmentator v2.0.1 masks; kidney cysts merged; no adrenal cleaning',
              'split_note': 'Dataset902 membership only; test does not establish absence from AtlasNet pretraining',
              'empty_reference': 'percent error null; Dice null only when both masks are empty',
              'inputs': {str(p): sha256(p) for p in (completion, build, splits, archive)},
              'code': {p.name: sha256(p) for p in (Path(__file__), Path(__file__).with_name('evaluate.py'))},
              'files': {p.name: sha256(p) for p in output.glob('*.csv')}}
    (output / 'complete.json').write_text(json.dumps(record, indent=2))
    print('COMPLETE: raw-reference scoring', flush=True)


if __name__ == '__main__':
    main()
