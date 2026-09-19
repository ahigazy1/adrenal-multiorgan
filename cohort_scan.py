"""Step 1: measure every scan and apply the adrenal cleaning rule. Changes no data.

For each case in TotalSegmentator v2, AMOS22 CT, BTCV and FLARE22 this writes one CSV row with the
volume (mL) of each target organ and, per adrenal gland, the connected-component analysis
and the resulting decision. The rule (see PLAN.md):

    gland absent                            -> absent             (scan kept)
    a secondary component >= 0.3 mL         -> exclude_fragmented (scan excluded)
    largest component <= 1 mL               -> exclude_small      (scan excluded)
    gland touches an edge of the scan       -> exclude_truncated  (scan excluded)
    secondary components all < 0.3 mL       -> keep_despeckled    (speckles removed later)
    exactly one component > 1 mL            -> keep

Usage (each source is optional):
    python cohort_scan.py --out results --ts Totalsegmentator_dataset_v201.zip --amos amos/ --btcv btcv/ --flare flare/
"""
import argparse
import csv
import gzip
import hashlib
import json
import logging
import sys
import zipfile
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import nibabel as nib
import numpy as np
import scipy
from scipy import ndimage

SPECKLE_ML = 0.3
MINIMUM_ML = 1.0
GLANDS = ('adrenal_left', 'adrenal_right')
OTHER_ORGANS = ('kidney_left', 'kidney_right', 'liver', 'spleen', 'aorta', 'inferior_vena_cava', 'pancreas')

# TotalSegmentator: one mask file per structure. Kidney cysts are merged into their kidney.
TS_FILES = {'adrenal_left': ['adrenal_gland_left'], 'adrenal_right': ['adrenal_gland_right'],
            'kidney_left': ['kidney_left', 'kidney_cyst_left'], 'kidney_right': ['kidney_right', 'kidney_cyst_right'],
            'liver': ['liver'], 'spleen': ['spleen'], 'aorta': ['aorta'],
            'inferior_vena_cava': ['inferior_vena_cava'], 'pancreas': ['pancreas']}
# AMOS, BTCV and FLARE22: one label file per case; these are the integer ids inside it.
LABEL_IDS = {
    'amos': {'spleen': 1, 'kidney_right': 2, 'kidney_left': 3, 'liver': 6, 'aorta': 8,
             'inferior_vena_cava': 9, 'pancreas': 10, 'adrenal_right': 11, 'adrenal_left': 12},
    'btcv': {'spleen': 1, 'kidney_right': 2, 'kidney_left': 3, 'liver': 6, 'aorta': 8,
             'inferior_vena_cava': 9, 'pancreas': 11, 'adrenal_right': 12, 'adrenal_left': 13},
    'flare': {'liver': 1, 'kidney_right': 2, 'spleen': 3, 'pancreas': 4, 'aorta': 5, 'inferior_vena_cava': 6,
              'adrenal_right': 7, 'adrenal_left': 8, 'kidney_left': 13}}
# Where each of those sources keeps its label files, and the split its own authors gave them.
LABEL_FOLDERS = (('amos', 'train/labelsTr', 'train'), ('amos', 'valid/labelsVa', 'val'),
                 ('btcv', 'RawData/Training/label', 'train'), ('flare', 'labels', 'train'))

log = logging.getLogger('cohort_scan')


def judge_gland(mask, voxel_ml):
    """Apply the cleaning rule to one gland. Returns (decision, component volumes in mL, largest first)."""
    components, count = ndimage.label(mask, structure=np.ones((3, 3, 3)))  # 26-connected
    if count == 0:
        return 'absent', []
    volumes = sorted((np.bincount(components.ravel())[1:] * voxel_ml).tolist(), reverse=True)
    if any(volume >= SPECKLE_ML for volume in volumes[1:]):
        return 'exclude_fragmented', volumes
    if volumes[0] <= MINIMUM_ML:
        return 'exclude_small', volumes
    if any(mask.take(index, axis=axis).any() for axis in range(3) for index in (0, -1)):
        return 'exclude_truncated', volumes  # cut by the edge of the scan: its volume is not the gland's volume
    return ('keep_despeckled' if count > 1 else 'keep'), volumes


def load_masks(source, case, official_split, folders):
    """Read one case's labels. Returns (label image, {organ: boolean mask}); folders = {'ts': zip, 'amos': dir, ...}."""
    if source == 'ts':
        with zipfile.ZipFile(folders['ts']) as archive:
            def read(structure):
                return nib.Nifti1Image.from_bytes(gzip.decompress(archive.read(f'{case}/segmentations/{structure}.nii.gz')))
            images = {s: read(s) for files in TS_FILES.values() for s in files}
            reference = images['liver']
            for structure, image in images.items():  # every mask must sit on the same grid
                if image.shape != reference.shape or not np.allclose(image.affine, reference.affine, atol=1e-3):
                    raise ValueError(f'{structure} is on a different grid than liver')
            masks = {organ: np.any([np.asanyarray(images[s].dataobj) > 0 for s in files], axis=0)
                     for organ, files in TS_FILES.items()}
            return reference, masks
    image = nib.load(label_path(source, case, official_split, folders))
    labels = np.asanyarray(image.dataobj)
    return image, {organ: labels == value for organ, value in LABEL_IDS[source].items()}


def check_sides(image, masks):
    """'ok', or which organs are on the wrong side of the body. Positions are taken in world coordinates, where
    nibabel's x axis points to the patient's right (RAS+), so this catches swapped left/right labels, a wrong orientation
    in the file header, and mirrored anatomy (situs inversus). Organs that are absent are not checked."""
    def world_x(organ):
        centre = ndimage.center_of_mass(masks[organ])
        return (image.affine @ [*centre, 1])[0]
    wrong = [f'{left} is not left of {right}'
             for left, right in (('adrenal_left', 'adrenal_right'), ('kidney_left', 'kidney_right'), ('spleen', 'liver'))
             if masks[left].any() and masks[right].any() and world_x(left) >= world_x(right)]
    return '; '.join(wrong) or 'ok'


def label_path(source, case, official_split, folders):
    folder = next(f for s, f, split in LABEL_FOLDERS if (s, split) == (source, official_split))
    return folders[source] / folder / f'{case}.nii.gz'


def voxel_volume_ml(image):
    return float(np.prod(image.header.get_zooms()[:3])) / 1000


def scan_case(job):
    """One CSV row. An unreadable case must not end a 1500-case scan: its error is recorded in the row instead."""
    source, case, official_split, folders = job
    row = {'source': source, 'case': case, 'official_split': official_split}
    try:
        image, masks = load_masks(*job)
        voxel_ml = voxel_volume_ml(image)
        row['voxel_ml'] = voxel_ml
        row['slice_thickness_mm'] = round(float(max(image.header.get_zooms()[:3])), 3)  # the coarsest axis
        for gland in GLANDS:
            decision, volumes = judge_gland(masks[gland], voxel_ml)
            row[gland] = decision
            row[gland + '_components'] = len(volumes)
            row[gland + '_largest_ml'] = round(volumes[0], 4) if volumes else 0
            row[gland + '_second_ml'] = round(volumes[1], 4) if len(volumes) > 1 else 0
        for organ in OTHER_ORGANS:
            row[organ + '_ml'] = round(float(masks[organ].sum()) * voxel_ml, 2)
        row['sides'] = check_sides(image, masks)
        decisions = {row[gland] for gland in GLANDS}
        if row['sides'] != 'ok' or any(d.startswith('exclude') for d in decisions):
            row['scan'] = 'excluded'
        elif decisions == {'absent'}:
            row['scan'] = 'kept_no_adrenal'
        else:
            row['scan'] = 'kept_with_adrenal'
        return row | {'error': ''}
    except Exception as error:
        return row | {'scan': 'error', 'error': repr(error)}


def write_csv(path, rows):
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with open(path, 'w', newline='') as stream:
        writer = csv.DictWriter(stream, columns, restval='')
        writer.writeheader()
        writer.writerows(rows)


def sha256(path):
    with open(path, 'rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out', required=True, type=Path, help='folder for cohort_scan.csv, .log and .json')
    parser.add_argument('--ts', type=Path, help='Totalsegmentator_dataset_v201.zip')
    parser.add_argument('--amos', type=Path, help='folder holding train/labelsTr and valid/labelsVa')
    parser.add_argument('--btcv', type=Path, help='folder holding RawData/Training/label')
    parser.add_argument('--flare', type=Path, help='folder holding labels/')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--limit', type=int, help='only the first N cases per source (for a quick test)')
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s',
                        handlers=[logging.StreamHandler(), logging.FileHandler(args.out / 'cohort_scan.log', 'w')])
    record = {'arguments': {k: str(v) for k, v in vars(args).items()}, 'speckle_ml': SPECKLE_ML,
              'minimum_ml': MINIMUM_ML, 'python': sys.version, 'numpy': np.__version__,
              'scipy': scipy.__version__, 'nibabel': nib.__version__, 'inputs': {}}

    folders = {'ts': args.ts, 'amos': args.amos, 'btcv': args.btcv, 'flare': args.flare}
    jobs = []
    if args.ts:
        log.info('Hashing %s', args.ts)
        record['inputs']['ts_zip_sha256'] = sha256(args.ts)
        with zipfile.ZipFile(args.ts) as archive:
            lines = archive.read('meta.csv').decode('utf-8-sig').splitlines()
        jobs += [('ts', m['image_id'], m['split'], folders) for m in csv.DictReader(lines, delimiter=';')][:args.limit]
    for source, folder, split in LABEL_FOLDERS:
        if folders[source]:
            files = sorted((folders[source] / folder).glob('*.nii.gz'))[:args.limit]
            jobs += [(source, f.name.removesuffix('.nii.gz'), split, folders) for f in files]
    log.info('Scanning %d cases with %d workers', len(jobs), args.workers)

    rows = []
    with Pool(args.workers) as pool:
        for row in pool.imap_unordered(scan_case, jobs, chunksize=4):
            rows.append(row)
            if row['error']:
                log.error('%s %s failed: %s', row['source'], row['case'], row['error'])
            if len(rows) % 100 == 0:
                log.info('%d of %d cases done', len(rows), len(jobs))

    rows.sort(key=lambda row: (row['source'], row['case']))
    write_csv(args.out / 'cohort_scan.csv', rows)

    counts = Counter(f"{row['source']} {row['official_split']} {row['scan']}" for row in rows)
    for key in sorted(counts):
        log.info('%-40s %d', key, counts[key])
    record['counts'] = counts
    # A wrong label id would show up here at once (a 5 mL "liver"), so log typical sizes per source.
    for source in sorted({row['source'] for row in rows}):
        for organ in GLANDS + OTHER_ORGANS:
            column = organ + ('_largest_ml' if organ in GLANDS else '_ml')
            present = [float(row[column]) for row in rows if row['source'] == source and float(row.get(column) or 0) > 0]
            log.info('%-5s %-20s present in %4d cases, median %8.1f mL', source, organ, len(present),
                     float(np.median(present)) if present else 0)
    record['csv_sha256'] = sha256(args.out / 'cohort_scan.csv')
    (args.out / 'cohort_scan.json').write_text(json.dumps(record, indent=2))
    log.info('Wrote %s', args.out / 'cohort_scan.csv')


def self_check():
    """Tiny synthetic examples of every branch of the rule, at 1 mm voxels (0.001 mL)."""
    gland = np.zeros((30, 30, 30), bool)
    assert judge_gland(gland, 0.001)[0] == 'absent'
    gland[1:11, 1:11, 1:11] = True  # exactly 1.0 mL is not > 1 mL
    assert judge_gland(gland, 0.001)[0] == 'exclude_small'
    gland[1:12, 1:11, 1:11] = True  # 1.1 mL
    assert judge_gland(gland, 0.001)[0] == 'keep'
    gland[20, 20, 20] = True  # one-voxel speckle
    assert judge_gland(gland, 0.001)[0] == 'keep_despeckled'
    gland[20:27, 20:27, 20:27] = True  # 0.343 mL second component
    assert judge_gland(gland, 0.001)[0] == 'exclude_fragmented'
    labels = np.zeros((30, 30, 30), np.uint8)  # multi-label file, as in AMOS
    labels[1:12, 1:11, 1:11] = LABEL_IDS['amos']['adrenal_left']
    assert judge_gland(labels == LABEL_IDS['amos']['adrenal_left'], 0.001)[0] == 'keep'
    assert judge_gland(labels == LABEL_IDS['amos']['adrenal_right'], 0.001)[0] == 'absent'
    assert all(set(ids) == set(GLANDS + OTHER_ORGANS) for ids in LABEL_IDS.values())  # every source maps all 9
    gland[20:27, 20:27, 20:27] = False  # back to one gland, then extend it to the edge of the volume
    gland[0, 5, 5] = True
    assert judge_gland(gland, 0.001)[0] == 'exclude_truncated'
    body = {organ: np.zeros((30, 30, 30), bool) for organ in GLANDS + OTHER_ORGANS}
    body['kidney_left'][5:10, 10:15, 10:15] = body['kidney_right'][20:25, 10:15, 10:15] = True  # x grows to the right
    ras, lps = nib.Nifti1Image(np.zeros((30, 30, 30)), np.eye(4)), nib.Nifti1Image(np.zeros((30, 30, 30)), np.diag([-1., -1, 1, 1]))
    assert check_sides(ras, body) == 'ok' and check_sides(lps, body) == 'kidney_left is not left of kidney_right'

if __name__ == '__main__':
    self_check()
    main()
