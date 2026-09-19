"""Step 2: write the cleaned nnU-Net dataset from the three sources. Reads step 1's cohort_scan.csv.

For every scan that step 1 kept, this writes
  - the CT, unchanged (bytes copied, never re-encoded)
  - one label file with the 9 classes of PLAN.md, with adrenal speckles removed
into nnUNet_raw/Dataset902_AdrenalMultiorgan, as imagesTr/labelsTr (training) or imagesTs/labelsTs
(held-out test), plus dataset.json, splits_final.json (fold 0) and build_dataset.csv (one row per case).

Who goes where:
  test             TotalSegmentator's official test split and AMOS's official validation set; BTCV and FLARE22
                   publish no labelled test set, so a seeded random 20% of each is held out instead
  train/validation everything else, divided by nnU-Net's own default split (5 folds, seed 12345); fold 0 is used

Each case is re-judged with step 1's rule and must get step 1's decision, so the two steps cannot drift apart.

Usage:
    python build_dataset.py --scan results/cohort_scan.csv --out data --ts TS.zip --amos amos/ --btcv btcv/ --flare flare/
"""
import argparse
import csv
import gzip
import json
import logging
import random
import sys
import zipfile
from multiprocessing import Pool
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage

from cohort_scan import GLANDS, judge_gland, load_masks, sha256, voxel_volume_ml, write_csv

sys.path.insert(0, str(Path(__file__).resolve().parent / 'vendor/nnUNet'))
from nnunetv2.utilities.crossval_split import generate_crossval_split

DATASET = 'Dataset902_AdrenalMultiorgan'
# Label values in the written files. Adrenals are written last so they win where TotalSegmentator masks overlap.
LABELS = {'background': 0, 'adrenal_left': 1, 'adrenal_right': 2, 'kidney_left': 3, 'kidney_right': 4,
          'liver': 5, 'spleen': 6, 'aorta': 7, 'inferior_vena_cava': 8, 'pancreas': 9}
WRITE_ORDER = ['liver', 'spleen', 'kidney_left', 'kidney_right', 'aorta', 'inferior_vena_cava', 'pancreas',
               'adrenal_left', 'adrenal_right']
HELD_OUT_FRACTION = 0.2  # of BTCV and of FLARE22
SEED = 12345  # the seed nnU-Net uses for its split

log = logging.getLogger('build_dataset')


def largest_component(mask):
    components, count = ndimage.label(mask, structure=np.ones((3, 3, 3)))  # 26-connected, as in step 1
    if count <= 1:
        return mask
    return components == 1 + np.argmax(np.bincount(components.ravel())[1:])


def image_path(source, case, official_split, folders):
    if source == 'amos':
        return folders['amos'] / ('train/imagesTr' if official_split == 'train' else 'valid/imagesVa') / f'{case}.nii.gz'
    if source == 'flare':
        return folders['flare'] / 'images' / f'{case}_0000.nii.gz'
    return folders['btcv'] / 'RawData/Training/img' / f"{case.replace('label', 'img')}.nii.gz"


def build_case(job):
    row, role, folders, out = job
    source, case = row['source'], row['case']
    name = case if case.lower().startswith(source) else f'{source}_{case}'  # ts_s0001, amos_0001, FLARE22_Tr_0001
    suffix = 'Ts' if role == 'test' else 'Tr'
    image_out = out / f'images{suffix}' / f'{name}_0000.nii.gz'
    label_out = out / f'labels{suffix}' / f'{name}.nii.gz'
    try:
        reference, masks = load_masks(source, case, row['official_split'], folders)
        if source == 'ts':
            with zipfile.ZipFile(folders['ts']) as archive:
                image_bytes = archive.read(f'{case}/ct.nii.gz')
        else:
            image_bytes = image_path(source, case, row['official_split'], folders).read_bytes()

        # The CT and its labels must describe the same grid.
        image = nib.Nifti1Image.from_bytes(gzip.decompress(image_bytes))
        if image.shape[:3] != reference.shape[:3] or not np.allclose(image.affine, reference.affine, atol=1e-3):
            raise ValueError(f'image grid {image.shape} and label grid {reference.shape} differ')

        voxel_ml = voxel_volume_ml(reference)
        removed_ml = {}
        for gland in GLANDS:
            decision, _ = judge_gland(masks[gland], voxel_ml)
            if decision != row[gland]:
                raise ValueError(f'{gland}: step 1 said {row[gland]}, step 2 says {decision}')
            cleaned = largest_component(masks[gland])
            removed_ml[gland] = round(float(masks[gland].sum() - cleaned.sum()) * voxel_ml, 4)
            masks[gland] = cleaned

        labels = np.zeros(reference.shape[:3], np.uint8)
        for organ in WRITE_ORDER:
            labels[masks[organ]] = LABELS[organ]
        overlap_voxels = int((np.sum(list(masks.values()), axis=0) > 1).sum())

        header = reference.header.copy()
        header.set_data_dtype(np.uint8)
        nib.save(nib.Nifti1Image(labels, reference.affine, header), label_out)
        image_out.write_bytes(image_bytes)
        return {'name': name, 'source': source, 'case': case, 'role': role,
                'speckle_removed_left_ml': removed_ml['adrenal_left'],
                'speckle_removed_right_ml': removed_ml['adrenal_right'],
                'overlap_voxels': overlap_voxels, 'labels_present': ' '.join(map(str, np.unique(labels)[1:])),
                'error': ''}
    except Exception as error:  # reported at the end; one bad file must not stop 1500 cases
        for path in (image_out, label_out):
            path.unlink(missing_ok=True)
        return {'name': name, 'source': source, 'case': case, 'role': 'failed', 'error': repr(error)}


def test_cases(rows):
    """(source, case) of every held-out scan among the kept rows."""
    held_out = {(r['source'], r['case']) for r in rows
                if (r['source'], r['official_split']) in (('ts', 'test'), ('amos', 'val'))}
    for source in ('btcv', 'flare'):
        cases = sorted(r['case'] for r in rows if r['source'] == source)  # sorted: the draw ignores input order
        held_out |= {(source, c) for c in random.Random(SEED).sample(cases, round(len(cases) * HELD_OUT_FRACTION))}
    return held_out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--scan', required=True, type=Path, help="step 1's cohort_scan.csv")
    parser.add_argument('--out', required=True, type=Path, help='data folder; nnUNet_raw and nnUNet_preprocessed go inside')
    parser.add_argument('--ts', type=Path)
    parser.add_argument('--amos', type=Path)
    parser.add_argument('--btcv', type=Path)
    parser.add_argument('--flare', type=Path)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--limit', type=int, help='only the first N kept cases per source (for a quick test)')
    args = parser.parse_args()

    raw = args.out / 'nnUNet_raw' / DATASET
    for folder in ('imagesTr', 'labelsTr', 'imagesTs', 'labelsTs'):
        (raw / folder).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s',
                        handlers=[logging.StreamHandler(), logging.FileHandler(raw / 'build_dataset.log', 'w')])

    folders = {'ts': args.ts, 'amos': args.amos, 'btcv': args.btcv, 'flare': args.flare}
    scanned = list(csv.DictReader(open(args.scan)))
    kept = [r for r in scanned if r['scan'].startswith('kept') and folders[r['source']]]
    log.info('%d scanned, %d kept by step 1 from the sources given, %d excluded or failed there',
             len(scanned), len(kept), len(scanned) - len(kept))
    held_out = test_cases(kept)  # before --limit, so a quick test gives each case the role it has in the full run
    if args.limit:
        kept = [r for s in folders for r in [k for k in kept if k['source'] == s][:args.limit]]

    rows = []
    with Pool(args.workers) as pool:
        jobs = [(r, 'test' if (r['source'], r['case']) in held_out else 'train', folders, raw) for r in kept]
        for row in pool.imap_unordered(build_case, jobs, chunksize=4):
            rows.append(row)
            if row['error']:
                log.error('%s failed: %s', row['name'], row['error'])
            if len(rows) % 100 == 0:
                log.info('%d of %d cases written', len(rows), len(kept))
    rows.sort(key=lambda r: r['name'])

    write_csv(raw / 'build_dataset.csv', rows)

    names = {role: [r['name'] for r in rows if r['role'] == role] for role in ('train', 'test', 'failed')}
    (raw / 'dataset.json').write_text(json.dumps({
        'channel_names': {'0': 'CT'}, 'labels': LABELS, 'file_ending': '.nii.gz', 'numTraining': len(names['train']),
        # AtlasNet's reader. Also needed because some TotalSegmentator CTs have orientation matrices that are
        # skewed by about 1e-4, which the default SimpleITK reader refuses to open.
        'overwrite_image_reader_writer': 'NibabelIOWithReorient'}, indent=2))
    # The split nnU-Net would create by itself at the first training run, written now so it is part of the dataset.
    splits = generate_crossval_split(names['train'])
    preprocessed = args.out / 'nnUNet_preprocessed' / DATASET
    preprocessed.mkdir(parents=True, exist_ok=True)
    (preprocessed / 'splits_final.json').write_text(json.dumps(splits, indent=2))
    log.info('Fold 0: %d training and %d validation cases', len(splits[0]['train']), len(splits[0]['val']))

    for source in folders:
        log.info('%-5s ' + '  '.join(f'{role} %4d' for role in names), source,
                 *[sum(r['source'] == source and r['role'] == role for r in rows) for role in names])
    record = {'scan_csv_sha256': sha256(args.scan), 'split': 'nnU-Net default: 5 folds, seed 12345',
              'held_out_fraction_btcv_flare': HELD_OUT_FRACTION, 'held_out_seed': SEED,
              'labels': LABELS, 'counts': {role: len(n) for role, n in names.items()},
              'build_dataset_csv_sha256': sha256(raw / 'build_dataset.csv')}
    (raw / 'build_dataset.json').write_text(json.dumps(record, indent=2))
    log.info('Wrote %s', raw)
    if names['failed']:
        raise SystemExit(f"{len(names['failed'])} cases failed; see build_dataset.log")


def self_check():
    gland = np.zeros((20, 20, 20), bool)
    gland[:5, :5, :5] = True
    gland[10, 10, 10] = True  # speckle
    cleaned = largest_component(gland)
    assert cleaned.sum() == 125 and not cleaned[10, 10, 10]
    rows = [{'source': 'btcv', 'case': f'label{i:04}', 'official_split': 'train'} for i in range(30)]
    rows += [{'source': 'ts', 'case': 's0001', 'official_split': 'val'}, {'source': 'ts', 'case': 's0002', 'official_split': 'test'},
             {'source': 'amos', 'case': 'amos_0001', 'official_split': 'val'}]
    held_out = test_cases(rows)
    assert len(held_out) == 8 and {('ts', 's0002'), ('amos', 'amos_0001')} <= held_out and ('ts', 's0001') not in held_out
    assert held_out == test_cases(rows[::-1])  # the draw does not depend on input order

if __name__ == '__main__':
    self_check()
    main()
