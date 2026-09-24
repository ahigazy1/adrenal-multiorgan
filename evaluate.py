"""Step 6: score a folder of predicted segmentations against the reference labels, on each scan's original grid.

    python evaluate.py data/nnUNet_raw/Dataset902_AdrenalMultiorgan/labelsTs data/predictions/final

Writes, next to the predictions:
    evaluation/per_case.csv     one row per scan and organ
    evaluation/all_organs.csv   summary per organ           evaluation/all_organs_by_source.csv
    evaluation/adrenal.csv      both adrenals only          evaluation/adrenal_by_source.csv
    evaluation/aorta.csv        aorta only                  evaluation/aorta_by_source.csv

Per scan and organ: Dice, normalised surface Dice (NSD) at 0.5, 0.8 and 1.0 mm (Google DeepMind's
`surface-distance` package), reference and predicted volume, and the volume error in mL
(prediction minus reference: negative means the model under-segments).
An organ absent from both reference and prediction is not scored (empty values). An organ present in only one
of them scores Dice 0 and NSD 0; present in the reference but not predicted also counts as a miss.
"""
import argparse
import json
import os
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import nibabel as nib
import numpy as np
import polars as pl
from surface_distance import metrics

from build_dataset import LABELS as ALL_LABELS

LABELS = {organ: value for organ, value in ALL_LABELS.items() if value}  # without the background
TOLERANCES_MM = [0.5, 0.8, 1.0]
REPORTS = {'all_organs': list(LABELS), 'adrenal': ['adrenal_left', 'adrenal_right'], 'aorta': ['aorta']}


def score_scan(paths, organs=tuple(LABELS)):
    reference_path, prediction_path = paths
    reference, prediction = nib.load(reference_path), nib.load(prediction_path)
    if reference.shape != prediction.shape or not np.allclose(reference.affine, prediction.affine, atol=1e-3):
        raise ValueError(f'{prediction_path.name}: prediction is not on the grid of the reference')
    truth, predicted = np.asanyarray(reference.dataobj), np.asanyarray(prediction.dataobj)
    spacing = nib.affines.voxel_sizes(reference.affine)
    voxel_ml = float(np.prod(spacing)) / 1000
    rows = []
    for organ in organs:
        value = LABELS[organ]
        a, b = truth == value, predicted == value
        in_truth, in_prediction = int(a.sum()), int(b.sum())
        row = {'case': reference_path.name[:-7], 'source': reference_path.name.split('_')[0], 'organ': organ,
               'reference_ml': in_truth * voxel_ml, 'predicted_ml': in_prediction * voxel_ml,
               'volume_error_ml': (in_prediction - in_truth) * voxel_ml,
               'absolute_volume_error_ml': abs(in_prediction - in_truth) * voxel_ml,
               'miss': in_truth > 0 and in_prediction == 0}
        if in_truth + in_prediction:
            row['dice'] = 2 * int((a & b).sum()) / (in_truth + in_prediction)
            both = in_truth and in_prediction
            distances = metrics.compute_surface_distances(a, b, spacing_mm=spacing) if both else None
            for tolerance in TOLERANCES_MM:
                row[f'nsd_{tolerance}mm'] = float(metrics.compute_surface_dice_at_tolerance(distances, tolerance)) if both else 0.0
        rows.append(row)
    return rows


def summarise(frame, groups):
    nsd = [f'nsd_{t}mm' for t in TOLERANCES_MM]
    return frame.group_by(groups).agg(
        pl.col('dice').count().alias('scored'), pl.col('miss').sum(),
        pl.col('dice').mean().alias('dice_mean'), pl.col('dice').median().alias('dice_median'),
        *[pl.col(n).mean().alias(f'{n}_mean') for n in nsd], *[pl.col(n).median().alias(f'{n}_median') for n in nsd],
        pl.col('volume_error_ml').mean().alias('volume_error_ml_mean'),
        pl.col('volume_error_ml').median().alias('volume_error_ml_median'),
        pl.col('absolute_volume_error_ml').mean().alias('absolute_volume_error_ml_mean'),
        pl.col('absolute_volume_error_ml').median().alias('absolute_volume_error_ml_median')).sort(groups)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('labels', type=Path)
    parser.add_argument('predictions', type=Path)
    parser.add_argument('--workers', type=int, default=os.cpu_count())
    parser.add_argument('--organs', nargs='+', choices=list(LABELS), default=list(LABELS),
                        help='only these (a test set that does not label every organ)')
    args = parser.parse_args()
    references = sorted(args.labels.glob('*.nii.gz'))
    missing = [r.name for r in references if not (args.predictions / r.name).exists()]
    if missing or not references:
        raise SystemExit(f'{len(missing)} of {len(references)} scans have no prediction, e.g. {missing[:3]}')
    with Pool(args.workers) as pool:
        rows = [row for scan in pool.imap(partial(score_scan, organs=args.organs),
                                          [(r, args.predictions / r.name) for r in references]) for row in scan]
    frame = pl.DataFrame(rows, infer_schema_length=None)
    output = args.predictions / 'evaluation'
    output.mkdir(exist_ok=True)
    frame.write_csv(output / 'per_case.csv')
    for name, organs in REPORTS.items():
        part = frame.filter(pl.col('organ').is_in(organs))
        if part.is_empty():
            continue
        summarise(part, ['organ']).write_csv(output / f'{name}.csv')
        summarise(part, ['source', 'organ']).write_csv(output / f'{name}_by_source.csv')
        print(f'\n=== {name} ===')
        with pl.Config(ascii_tables=True, tbl_cols=-1, tbl_rows=-1, tbl_width_chars=250):
            print(summarise(part, ['organ']))
    (output / 'settings.json').write_text(json.dumps({
        'scans': len(references), 'labels': {organ: LABELS[organ] for organ in args.organs}, 'nsd_tolerances_mm': TOLERANCES_MM, 'nsd_package': 'surface-distance 0.1',
        'volume_error': 'prediction minus reference, mL', 'absent_in_both': 'not scored', 'missed_organ': 'Dice 0, NSD 0'}, indent=2))


def self_check():
    a = np.zeros((20, 20, 20), bool); a[5:15, 5:15, 5:15] = True
    d = metrics.compute_surface_distances(a, a, spacing_mm=(1, 1, 1))
    assert metrics.compute_surface_dice_at_tolerance(d, 0.5) == 1


if __name__ == '__main__':
    self_check()
    main()
