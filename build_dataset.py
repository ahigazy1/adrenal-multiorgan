"""Step 2: write the cleaned nnU-Net dataset from the three sources. Reads step 1's cohort_scan.csv.

For every scan that step 1 kept, this writes
  - the CT, unchanged (bytes copied, never re-encoded)
  - one label file with the 9 classes of PLAN.md, with adrenal speckles removed
into nnUNet_raw/Dataset902_AdrenalMultiorgan, as imagesTr/labelsTr (training) or imagesTs/labelsTs
(held-out test), plus dataset.json, splits_final.json (fold 0) and build_dataset.csv (one row per case).

Who goes where:
  test        TotalSegmentator's official test split, AMOS's official validation set, 6 fixed BTCV scans
  validation  a seeded random 20% of the remaining scans of each source (used only to monitor training)
  train       everything else

Each case is re-judged with step 1's rule and must get step 1's decision, so the two steps cannot drift apart.

Usage:
    python build_dataset.py --scan results/cohort_scan.csv --out data --ts TS.zip --amos amos/ --btcv btcv/
"""
import argparse
import csv
import gzip
import json
import logging
import random
import zipfile
from multiprocessing import Pool
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage

from cohort_scan import AMOS_IDS, BTCV_IDS, GLANDS, TS_FILES, judge_gland, sha256, voxel_volume_ml

DATASET = 'Dataset902_AdrenalMultiorgan'
# Label values in the written files. Adrenals are written last so they win where TotalSegmentator masks overlap.
LABELS = {'background': 0, 'adrenal_left': 1, 'adrenal_right': 2, 'kidney_left': 3, 'kidney_right': 4,
          'liver': 5, 'spleen': 6, 'aorta': 7, 'inferior_vena_cava': 8, 'pancreas': 9}
WRITE_ORDER = ['liver', 'spleen', 'kidney_left', 'kidney_right', 'aorta', 'inferior_vena_cava', 'pancreas',
               'adrenal_left', 'adrenal_right']
BTCV_TEST = {'label0001', 'label0004', 'label0008', 'label0009', 'label0031', 'label0034'}  # as in the pilot study
VALIDATION_FRACTION = 0.2
SEED = 42

log = logging.getLogger('build_dataset')


def largest_component(mask):
    components, count = ndimage.label(mask, structure=np.ones((3, 3, 3)))  # 26-connected, as in step 1
    if count <= 1:
        return mask
    return components == 1 + np.argmax(np.bincount(components.ravel())[1:])


def build_case(job):
    row, role, sources, out = job
    source, case = row['source'], row['case']
    name = f'{source}_{case}'
    suffix = 'Ts' if role == 'test' else 'Tr'
    image_out = out / f'images{suffix}' / f'{name}_0000.nii.gz'
    label_out = out / f'labels{suffix}' / f'{name}.nii.gz'
    try:
        if source == 'ts':
            with zipfile.ZipFile(sources['ts']) as archive:
                def read(structure):
                    data = gzip.decompress(archive.read(f'{case}/segmentations/{structure}.nii.gz'))
                    return nib.Nifti1Image.from_bytes(data)
                reference = read('liver')
                masks = {organ: np.any([np.asanyarray(read(s).dataobj) > 0 for s in files], axis=0)
                         for organ, files in TS_FILES.items()}
                image_bytes = archive.read(f'{case}/ct.nii.gz')
        else:
            if source == 'amos':
                folder = 'train' if row['official_split'] == 'train' else 'valid'
                end = 'Tr' if folder == 'train' else 'Va'
                label_path = sources['amos'] / folder / f'labels{end}' / f'{case}.nii.gz'
                image_path = sources['amos'] / folder / f'images{end}' / f'{case}.nii.gz'
                ids = AMOS_IDS
            else:
                label_path = sources['btcv'] / 'RawData/Training/label' / f'{case}.nii.gz'
                image_path = sources['btcv'] / 'RawData/Training/img' / f"{case.replace('label', 'img')}.nii.gz"
                ids = BTCV_IDS
            reference = nib.load(label_path)
            original = np.asanyarray(reference.dataobj)
            masks = {organ: original == ids[organ] for organ in LABELS if organ != 'background'}
            image_bytes = image_path.read_bytes()

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


def assign_roles(rows):
    """Test sets are fixed by the datasets themselves; validation is a seeded 20% of the rest, per source."""
    roles = {}
    for source in ('ts', 'amos', 'btcv'):
        rest = []
        for row in (r for r in rows if r['source'] == source):
            is_test = (row['official_split'] == 'test' if source == 'ts' else
                       row['official_split'] == 'val' if source == 'amos' else row['case'] in BTCV_TEST)
            if is_test:
                roles[source, row['case']] = 'test'
            else:
                rest.append(row['case'])
        rest.sort()
        random.Random(SEED).shuffle(rest)
        cut = round(len(rest) * VALIDATION_FRACTION)
        roles |= {(source, case): 'validation' for case in rest[:cut]}
        roles |= {(source, case): 'train' for case in rest[cut:]}
    return roles


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--scan', required=True, type=Path, help="step 1's cohort_scan.csv")
    parser.add_argument('--out', required=True, type=Path, help='data folder; nnUNet_raw and nnUNet_preprocessed go inside')
    parser.add_argument('--ts', type=Path)
    parser.add_argument('--amos', type=Path)
    parser.add_argument('--btcv', type=Path)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--limit', type=int, help='only the first N kept cases per source (for a quick test)')
    args = parser.parse_args()

    raw = args.out / 'nnUNet_raw' / DATASET
    for folder in ('imagesTr', 'labelsTr', 'imagesTs', 'labelsTs'):
        (raw / folder).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s',
                        handlers=[logging.StreamHandler(), logging.FileHandler(raw / 'build_dataset.log', 'w')])

    sources = {'ts': args.ts, 'amos': args.amos, 'btcv': args.btcv}
    scanned = list(csv.DictReader(open(args.scan)))
    kept = [r for r in scanned if r['scan'].startswith('kept') and sources[r['source']]]
    log.info('%d scanned, %d kept by step 1 from the sources given, %d excluded or failed there',
             len(scanned), len(kept), len(scanned) - len(kept))
    roles = assign_roles(kept)  # before --limit, so a test run gives each case the role it has in the full run
    if args.limit:
        kept = [r for s in sources for r in [k for k in kept if k['source'] == s][:args.limit]]

    rows = []
    with Pool(args.workers) as pool:
        jobs = [(r, roles[r['source'], r['case']], sources, raw) for r in kept]
        for row in pool.imap_unordered(build_case, jobs, chunksize=4):
            rows.append(row)
            if row['error']:
                log.error('%s failed: %s', row['name'], row['error'])
            if len(rows) % 100 == 0:
                log.info('%d of %d cases written', len(rows), len(kept))
    rows.sort(key=lambda r: r['name'])

    columns = list(dict.fromkeys(key for row in rows for key in row))
    with open(raw / 'build_dataset.csv', 'w', newline='') as stream:
        writer = csv.DictWriter(stream, columns, restval='')
        writer.writeheader()
        writer.writerows(rows)

    names = {role: [r['name'] for r in rows if r['role'] == role] for role in ('train', 'validation', 'test', 'failed')}
    (raw / 'dataset.json').write_text(json.dumps({
        'channel_names': {'0': 'CT'}, 'labels': LABELS, 'file_ending': '.nii.gz',
        # AtlasNet's reader. Also needed because some TotalSegmentator CTs have orientation matrices that are
        # skewed by about 1e-4, which the default SimpleITK reader refuses to open.
        'overwrite_image_reader_writer': 'NibabelIOWithReorient',
        'numTraining': len(names['train']) + len(names['validation'])}, indent=2))
    preprocessed = args.out / 'nnUNet_preprocessed' / DATASET
    preprocessed.mkdir(parents=True, exist_ok=True)
    (preprocessed / 'splits_final.json').write_text(json.dumps(
        [{'train': names['train'], 'val': names['validation']}], indent=2))  # nnU-Net reads entry 0 as fold 0

    for source in ('ts', 'amos', 'btcv'):
        log.info('%-5s ' + '  '.join(f'{role} %4d' for role in names), source,
                 *[sum(r['source'] == source and r['role'] == role for r in rows) for role in names])
    record = {'scan_csv_sha256': sha256(args.scan), 'seed': SEED, 'validation_fraction': VALIDATION_FRACTION,
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
    rows = [{'source': 'ts', 'case': f's{i:04}', 'official_split': 'test' if i < 10 else 'train'} for i in range(110)]
    rows += [{'source': 'btcv', 'case': f'label{i:04}', 'official_split': 'train'} for i in range(1, 11)]
    roles = assign_roles(rows)
    count = lambda source, role: sum(r == role for (s, _), r in roles.items() if s == source)
    assert (count('ts', 'test'), count('ts', 'validation'), count('ts', 'train')) == (10, 20, 80)
    assert (count('btcv', 'test'), count('btcv', 'validation'), count('btcv', 'train')) == (4, 1, 5)
    assert roles == assign_roles(rows[::-1])  # the split does not depend on input order


if __name__ == '__main__':
    self_check()
    main()
