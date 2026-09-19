"""Step 1: measure every scan and apply the adrenal cleaning rule. Changes no data.

For each case in TotalSegmentator v2, AMOS22 CT and BTCV this writes one CSV row with the
volume (mL) of each target organ and, per adrenal gland, the connected-component analysis
and the resulting decision. The rule (see PLAN.md):

    gland absent                            -> absent             (scan kept)
    a secondary component >= 0.3 mL         -> exclude_fragmented (scan excluded)
    largest component <= 1 mL               -> exclude_small      (scan excluded)
    secondary components all < 0.3 mL       -> keep_despeckled    (speckles removed later)
    exactly one component > 1 mL            -> keep

Usage (each source is optional):
    python cohort_scan.py --out results --ts Totalsegmentator_dataset_v201.zip --amos amos/ --btcv btcv/
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
# AMOS and BTCV: one label file per case; these are the integer ids inside it.
AMOS_IDS = {'spleen': 1, 'kidney_right': 2, 'kidney_left': 3, 'liver': 6, 'aorta': 8,
            'inferior_vena_cava': 9, 'pancreas': 10, 'adrenal_right': 11, 'adrenal_left': 12}
BTCV_IDS = {'spleen': 1, 'kidney_right': 2, 'kidney_left': 3, 'liver': 6, 'aorta': 8,
            'inferior_vena_cava': 9, 'pancreas': 11, 'adrenal_right': 12, 'adrenal_left': 13}

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
    return ('keep_despeckled' if count > 1 else 'keep'), volumes


def load_masks(source, case, official_split, folders):
    """Read one case's labels. Returns (label image, {organ: boolean mask}); folders = {'ts': zip, 'amos': dir, 'btcv': dir}."""
    if source == 'ts':
        with zipfile.ZipFile(folders['ts']) as archive:
            def read(structure):
                return nib.Nifti1Image.from_bytes(gzip.decompress(archive.read(f'{case}/segmentations/{structure}.nii.gz')))
            masks = {organ: np.any([np.asanyarray(read(s).dataobj) > 0 for s in files], axis=0)
                     for organ, files in TS_FILES.items()}
            return read('liver'), masks
    image = nib.load(label_path(source, case, official_split, folders))
    labels = np.asanyarray(image.dataobj)
    return image, {organ: labels == value for organ, value in (AMOS_IDS if source == 'amos' else BTCV_IDS).items()}


def label_path(source, case, official_split, folders):
    if source == 'amos':
        return folders['amos'] / ('train/labelsTr' if official_split == 'train' else 'valid/labelsVa') / f'{case}.nii.gz'
    return folders['btcv'] / 'RawData/Training/label' / f'{case}.nii.gz'


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
        for gland in GLANDS:
            decision, volumes = judge_gland(masks[gland], voxel_ml)
            row[gland] = decision
            row[gland + '_components'] = len(volumes)
            row[gland + '_largest_ml'] = round(volumes[0], 4) if volumes else 0
            row[gland + '_second_ml'] = round(volumes[1], 4) if len(volumes) > 1 else 0
        for organ in OTHER_ORGANS:
            row[organ + '_ml'] = round(float(masks[organ].sum()) * voxel_ml, 2)
        decisions = {row[gland] for gland in GLANDS}
        if any(d.startswith('exclude') for d in decisions):
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
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--limit', type=int, help='only the first N cases per source (for a quick test)')
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s',
                        handlers=[logging.StreamHandler(), logging.FileHandler(args.out / 'cohort_scan.log', 'w')])
    record = {'arguments': {k: str(v) for k, v in vars(args).items()}, 'speckle_ml': SPECKLE_ML,
              'minimum_ml': MINIMUM_ML, 'python': sys.version, 'numpy': np.__version__,
              'scipy': scipy.__version__, 'nibabel': nib.__version__, 'inputs': {}}

    folders = {'ts': args.ts, 'amos': args.amos, 'btcv': args.btcv}
    jobs = []
    if args.ts:
        log.info('Hashing %s', args.ts)
        record['inputs']['ts_zip_sha256'] = sha256(args.ts)
        with zipfile.ZipFile(args.ts) as archive:
            lines = archive.read('meta.csv').decode('utf-8-sig').splitlines()
        jobs += [('ts', m['image_id'], m['split'], folders) for m in csv.DictReader(lines, delimiter=';')][:args.limit]
    for source, folder, split in (('amos', 'train/labelsTr', 'train'), ('amos', 'valid/labelsVa', 'val'),
                                  ('btcv', 'RawData/Training/label', 'train')):
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
    gland[:10, :10, :10] = True  # exactly 1.0 mL is not > 1 mL
    assert judge_gland(gland, 0.001)[0] == 'exclude_small'
    gland[:11, :10, :10] = True  # 1.1 mL
    assert judge_gland(gland, 0.001)[0] == 'keep'
    gland[20, 20, 20] = True  # one-voxel speckle
    assert judge_gland(gland, 0.001)[0] == 'keep_despeckled'
    gland[20:27, 20:27, 20:27] = True  # 0.343 mL second component
    assert judge_gland(gland, 0.001)[0] == 'exclude_fragmented'
    labels = np.zeros((30, 30, 30), np.uint8)  # multi-label file, as in AMOS
    labels[:11, :10, :10] = AMOS_IDS['adrenal_left']
    assert judge_gland(labels == AMOS_IDS['adrenal_left'], 0.001)[0] == 'keep'
    assert judge_gland(labels == AMOS_IDS['adrenal_right'], 0.001)[0] == 'absent'

if __name__ == '__main__':
    self_check()
    main()
