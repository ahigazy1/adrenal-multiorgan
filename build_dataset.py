"""Step 2: write the cleaned nnU-Net dataset from the three sources. Reads step 1's cohort_scan.csv.

For every scan that step 1 kept, this writes
  - the CT, unchanged (bytes copied, never re-encoded)
  - one label file with the 9 classes of PLAN.md, with adrenal speckles removed
into nnUNet_raw/Dataset902_AdrenalMultiorgan, as imagesTr/labelsTr (training) or imagesTs/labelsTs
(held-out test), plus dataset.json, splits_final.json (fold 0) and build_dataset.csv (one row per case).

Who goes where. No dataset's published split is used. Within each source, scans are grouped by slice thickness
and by total adrenal volume (see stratum()), and scikit-learn's StratifiedKFold divides every group evenly:
  test        one fifth of all kept scans
  validation  one fifth of the rest (fold 0 of five; all five folds are written, as nnU-Net expects)
  train       everything else
The mix of each part is logged and saved, so the balance can be checked rather than assumed.

Each case is re-judged with step 1's rule and must get step 1's decision, so the two steps cannot drift apart.

Usage:
    python build_dataset.py --scan results/cohort_scan.csv --out data --ts TS.zip --amos amos/ --btcv btcv/ --flare flare/
"""
import argparse
import csv
import gzip
import io
import json
import logging
import sys
import zipfile
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage
from sklearn.model_selection import StratifiedKFold

from cohort_scan import GLANDS, judge_gland, load_masks, open_zip, sha256, voxel_volume_ml, write_csv

DATASET = 'Dataset902_AdrenalMultiorgan'
# Label values in the written files. Adrenals are written last so they win where TotalSegmentator masks overlap.
LABELS = {'background': 0, 'adrenal_left': 1, 'adrenal_right': 2, 'kidney_left': 3, 'kidney_right': 4,
          'liver': 5, 'spleen': 6, 'aorta': 7, 'inferior_vena_cava': 8, 'pancreas': 9}
WRITE_ORDER = ['liver', 'spleen', 'kidney_left', 'kidney_right', 'aorta', 'inferior_vena_cava', 'pancreas',
               'adrenal_left', 'adrenal_right']
SEED = 12345  # the seed nnU-Net uses for its own splits
MINIMUM_GROUP = 10  # smallest stratum that is still divided separately

log = logging.getLogger('build_dataset')
# nibabel warns once per file about a header field (qfac) that some source files leave at 0; it then uses the
# standard value, as nnU-Net's reader does. Only real errors from nibabel are shown.
logging.getLogger('nibabel').setLevel(logging.ERROR)


def largest_component(mask):
    components, count = ndimage.label(mask, structure=np.ones((3, 3, 3)))  # 26-connected, as in step 1
    if count <= 1:
        return mask
    return components == 1 + np.argmax(np.bincount(components.ravel())[1:])


def case_name(row):
    source, case = row['source'], row['case']
    return case if case.lower().startswith(source) else f'{source}_{case}'  # ts_s0001, amos_0001, FLARE22_Tr_0001


def image_path(source, case, official_split, folders):
    if source == 'amos':
        return folders['amos'] / ('train/imagesTr' if official_split == 'train' else 'valid/imagesVa') / f'{case}.nii.gz'
    if source == 'flare':
        return folders['flare'] / 'images' / f'{case}_0000.nii.gz'
    return folders['btcv'] / 'RawData/Training/img' / f"{case.replace('label', 'img')}.nii.gz"


def build_case(job):
    row, role, folders, out = job
    source, case = row['source'], row['case']
    name = case_name(row)
    suffix = 'Ts' if role == 'test' else 'Tr'
    image_out = out / f'images{suffix}' / f'{name}_0000.nii.gz'
    label_out = out / f'labels{suffix}' / f'{name}.nii.gz'
    try:
        reference, masks = load_masks(source, case, row['official_split'], folders)
        if source == 'ts':
            image_bytes = open_zip(folders['ts']).read(f'{case}/ct.nii.gz')
        else:
            image_bytes = image_path(source, case, row['official_split'], folders).read_bytes()

        # The CT and its labels must describe the same grid. The label file is then written with the CT's own
        # affine, so a reader that reorients by header (nnU-Net's NibabelIOWithReorient) treats both identically.
        # Only the CT's header is needed for this, so only its first bytes are decompressed.
        header_bytes = gzip.GzipFile(fileobj=io.BytesIO(image_bytes)).read(348)
        ct = nib.Nifti1Header.from_fileobj(io.BytesIO(header_bytes))
        ct_shape, ct_affine = ct.get_data_shape(), ct.get_best_affine()
        if ct_shape[:3] != reference.shape[:3] or not np.allclose(ct_affine, reference.affine, atol=1e-3):
            raise ValueError(f'image grid {ct_shape} and label grid {reference.shape} differ')

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
        covered = np.zeros(reference.shape[:3], np.uint8)  # how many organ masks claim each voxel
        for mask in masks.values():
            covered += mask
        overlap_voxels = int((covered > 1).sum())

        header = reference.header.copy()
        header.set_data_dtype(np.uint8)
        nib.save(nib.Nifti1Image(labels, ct_affine, header), label_out)
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


def thickness_group(row):
    thickness = float(row['slice_thickness_mm'])
    return '<=2 mm' if thickness <= 2 else '2-4 mm' if thickness <= 4 else '4-5 mm'


def stratum_labels(rows):
    """One grouping label per scan: source | slice thickness | total adrenal volume (thirds of the scans that have adrenals).
    A group too small to spread over five folds falls back to source | volume, then to the source alone."""
    volumes = [float(r['adrenal_left_largest_ml']) + float(r['adrenal_right_largest_ml']) for r in rows]
    low, high = np.quantile([v for v in volumes if v > 0], [1 / 3, 2 / 3])
    levels = []
    for row, volume in zip(rows, volumes):
        size = 'none' if volume == 0 else 'small' if volume <= low else 'medium' if volume <= high else 'large'
        levels.append((f"{row['source']}|{thickness_group(row)}|{size}", f"{row['source']}|{size}", row['source']))
    counts = Counter(label for level in levels for label in level)
    return [next((label for label in level if counts[label] >= MINIMUM_GROUP), level[-1]) for level in levels]


def five_folds(rows):
    """[(indices of four fifths, indices of one fifth), ...], every stratum spread evenly over the fifths."""
    rows = sorted(rows, key=lambda r: (r['source'], r['case']))  # the result must not depend on input order
    folds = StratifiedKFold(5, shuffle=True, random_state=SEED).split(rows, stratum_labels(rows))
    return [([rows[i] for i in most], [rows[i] for i in fifth]) for most, fifth in folds]


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
    development, test = five_folds(kept)[0]  # before --limit, so a quick test gives each case its full-run role
    held_out = {(r['source'], r['case']) for r in test}
    folds = five_folds(development)
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
    written = set(names['train'])
    splits = [{'train': [n for n in map(case_name, most) if n in written], 'val': [n for n in map(case_name, fifth) if n in written]}
              for most, fifth in folds]
    preprocessed = args.out / 'nnUNet_preprocessed' / DATASET
    preprocessed.mkdir(parents=True, exist_ok=True)
    (preprocessed / 'splits_final.json').write_text(json.dumps(splits, indent=2))
    log.info('Fold 0: %d training and %d validation cases', len(splits[0]['train']), len(splits[0]['val']))

    for source in folders:
        log.info('%-5s ' + '  '.join(f'{role} %4d' for role in names), source,
                 *[sum(r['source'] == source and r['role'] == role for r in rows) for role in names])
    # The balance of each part, over all kept scans (not only those written when --limit is used).
    parts = {'test': test, 'validation (fold 0)': folds[0][1], 'train (fold 0)': folds[0][0]}
    label_of = dict(zip(map(case_name, kept), stratum_labels(kept)))
    mix = {part: dict(sorted(Counter(label_of[case_name(r)] for r in members).items())) for part, members in parts.items()}
    for part, members in parts.items():
        volumes = [float(r['adrenal_left_largest_ml']) + float(r['adrenal_right_largest_ml']) for r in members]
        present = [v for v in volumes if v > 0]
        log.info('%-20s %4d scans | with adrenals %4d, median total adrenal volume %.2f mL | slice thickness: %s', part,
                 len(members), len(present), float(np.median(present)) if present else 0,
                 dict(sorted(Counter(map(thickness_group, members)).items())))
    record = {'scan_csv_sha256': sha256(args.scan), 'seed': SEED,
              'split': 'StratifiedKFold(5): fold 0 of all kept scans is the test set; the rest is divided again for training',
              'scans_per_stratum': mix,
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
    rows = [{'source': source, 'case': f'{source}_{thickness}_{i:03}', 'slice_thickness_mm': thickness,
             'adrenal_left_largest_ml': i % 7, 'adrenal_right_largest_ml': i % 5}
            for source, thickness, count in (('ts', 1.5, 200), ('amos', 5.0, 60), ('amos', 1.25, 40), ('btcv', 3.0, 30))
            for i in range(count)]
    development, test = five_folds(rows)[0]
    assert len(test) == 66 and len(development) == 264
    share = lambda members, source: sum(r['source'] == source for r in members) / len(members)
    assert all(abs(share(test, s) - share(rows, s)) < 0.01 for s in ('ts', 'amos', 'btcv'))  # every source is one fifth
    thick = lambda members: sum(r['slice_thickness_mm'] == 5.0 for r in members)
    assert abs(thick(test) - 12) <= 1  # one fifth of the 60 thick-slice scans, up to rounding within strata
    assert [r['case'] for r in test] == [r['case'] for r in five_folds(rows[::-1])[0][1]]  # input order is irrelevant

if __name__ == '__main__':
    self_check()
    main()
