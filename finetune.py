"""D997 fine-tune data: Dataset903 in D997's 66 classes, preprocessed with D997's own plans. Then `ADRENAL_RUN=d997 python train.py`.

    python finetune.py build --data data --scan results/cohort_scan.csv --pseudo /opt/pseudo/pseudolabels/totalseg-2.18.0/bfece4a56f11d1f1
    python finetune.py preprocess --data data

Labels. TotalSegmentator scans: their own masks (kidney cysts merged into the kidneys). AMOS, BTCV, FLARE22: every organ
the source labels is authoritative everywhere in the scan; TotalSegmentator 2.18 pseudo-labels (pseudolabels/) fill
the other classes. Adrenal speckles are removed as in build_dataset.py.

Exclusion, revised from PLAN.md for segmentation training: a scan is dropped only if an adrenal is fragmented, or is
fully in view (touches no scan face) and <= 1 mL, or the left/right check failed, or slices are > 5 mm. Glands cut by
the scan edge are correctly labelled partial glands and are kept.

Split. Test: one fifth of the AMOS/BTCV/FLARE scans, stratified as build_dataset.py does; never TotalSegmentator, which
D997 was trained on. Validation: fold 0 of five over everything else.
"""
import argparse
import csv
import gzip
import io
import json
import logging
import os
import shutil
import sys
from multiprocessing import Pool
from pathlib import Path

import nibabel as nib
import numpy as np

from build_dataset import case_name, five_folds, image_path, largest_component
from cohort_scan import judge_gland, label_path, open_zip, sha256, voxel_volume_ml, write_csv

DATASET = 'Dataset903_D997Finetune'
PLANS = 'D997Plans'
D997 = {'repo': 'ahigazy1/adrenal-training-models', 'revision': '3a52ec5a44db23a5d98375a6a21bfbcbe92a3dbe',
        'folder': 'labmate', 'plans': 'nnUNetResEncUNetLPlans', 'source_dataset': 'Dataset997_ChestAbd',
        'checkpoint_sha256': '0840b5b15364daa194033cef1466b522316c440cc190c80fdf7091dca02c17f7'}
GLANDS = ('adrenal_gland_left', 'adrenal_gland_right')
CYSTS = {'kidney_cyst_left': 'kidney_left', 'kidney_cyst_right': 'kidney_right'}
# Each source's own label ids -> D997 class names (AMOS 15 prostate/uterus has no D997 class; pseudo-labels cover it).
GROUND_TRUTH = {
    'amos': {1: 'spleen', 2: 'kidney_right', 3: 'kidney_left', 4: 'gallbladder', 5: 'esophagus', 6: 'liver', 7: 'stomach',
             8: 'aorta', 9: 'inferior_vena_cava', 10: 'pancreas', 11: 'adrenal_gland_right', 12: 'adrenal_gland_left',
             13: 'duodenum', 14: 'urinary_bladder'},
    'btcv': {1: 'spleen', 2: 'kidney_right', 3: 'kidney_left', 4: 'gallbladder', 5: 'esophagus', 6: 'liver', 7: 'stomach',
             8: 'aorta', 9: 'inferior_vena_cava', 10: 'portal_vein_and_splenic_vein', 11: 'pancreas',
             12: 'adrenal_gland_right', 13: 'adrenal_gland_left'},
    'flare': {1: 'liver', 2: 'kidney_right', 3: 'spleen', 4: 'pancreas', 5: 'aorta', 6: 'inferior_vena_cava',
              7: 'adrenal_gland_right', 8: 'adrenal_gland_left', 9: 'gallbladder', 10: 'esophagus', 11: 'stomach',
              12: 'duodenum', 13: 'kidney_left'}}

log = logging.getLogger('finetune')
logging.getLogger('nibabel').setLevel(logging.ERROR)


def fetch_d997(data):
    """D997's dataset.json, plans and final checkpoint at the pinned revision; the checkpoint's checksum is verified."""
    from huggingface_hub import hf_hub_download
    target = data / 'd997'
    for member in ('dataset.json', 'plans.json', 'fold_0/checkpoint_final.pth'):
        hf_hub_download(D997['repo'], f"{D997['folder']}/{member}", revision=D997['revision'], local_dir=target)
    folder = target / D997['folder']
    if sha256(folder / 'fold_0/checkpoint_final.pth') != D997['checkpoint_sha256']:
        raise SystemExit('D997 checkpoint failed its pinned checksum')
    return folder


def training_exclusion(masks, voxel_ml):
    """'' to keep, or why the scan is excluded (see the module docstring)."""
    for gland, mask in masks.items():
        decision, _ = judge_gland(mask, voxel_ml)
        touches_edge = any(mask.take(index, axis=axis).any() for axis in range(3) for index in (0, -1))
        if decision == 'exclude_fragmented' or (decision == 'exclude_small' and not touches_edge):
            return f'{gland}: {decision}'
    return ''


def build_case(job):
    row, role, folders, pseudo, labels, out = job
    source, case = row['source'], row['case']
    name = case_name(row)
    suffix = 'Ts' if role == 'test' else 'Tr'
    image_out, label_out = out / f'images{suffix}' / f'{name}_0000.nii.gz', out / f'labels{suffix}' / f'{name}.nii.gz'
    try:
        if source == 'ts':
            archive = open_zip(folders['ts'])
            read = lambda s: nib.Nifti1Image.from_bytes(gzip.decompress(archive.read(f'{case}/segmentations/{s}.nii.gz')))
            reference = read('liver')
            image_bytes = archive.read(f'{case}/ct.nii.gz')
            label = np.zeros(reference.shape[:3], np.uint8)
            kidney_cyst = {kidney: cyst for cyst, kidney in CYSTS.items()}
            for organ in labels:  # one mask at a time: 68 full-size masks per worker would not fit 24 workers in RAM
                if organ != 'background' and organ not in GLANDS:
                    mask = np.asanyarray(read(organ).dataobj) > 0
                    if organ in kidney_cyst:
                        mask |= np.asanyarray(read(kidney_cyst[organ]).dataobj) > 0
                    label[mask] = labels[organ]
            truth = {g: np.asanyarray(read(g).dataobj) > 0 for g in GLANDS}
        else:
            reference = nib.load(label_path(source, case, row['official_split'], folders))
            image_bytes = image_path(source, case, row['official_split'], folders).read_bytes()
            source_labels = np.asanyarray(reference.dataobj)
            teacher = nib.load(pseudo / source / f'{case}.nii.gz')  # pseudo-labels are named like the source files
            if teacher.shape != reference.shape or not np.allclose(teacher.affine, reference.affine, atol=1e-3):
                raise ValueError('pseudo-label grid differs from the label grid')
            label = np.asanyarray(teacher.dataobj).astype(np.uint8)
            truth = {organ: source_labels == value for value, organ in GROUND_TRUTH[source].items()}
            label[np.isin(label, [labels[organ] for organ in truth])] = 0  # the source is authoritative for its organs

        # The CT and its labels must describe the same grid (only the CT header is decompressed for this).
        ct = nib.Nifti1Header.from_fileobj(io.BytesIO(gzip.GzipFile(fileobj=io.BytesIO(image_bytes)).read(348)))
        if ct.get_data_shape()[:3] != reference.shape[:3] or not np.allclose(ct.get_best_affine(), reference.affine, atol=1e-3):
            raise ValueError('image grid and label grid differ')

        voxel_ml = voxel_volume_ml(reference)
        excluded = training_exclusion({g: truth[g] for g in GLANDS}, voxel_ml)
        if excluded:
            return {'name': name, 'source': source, 'case': case, 'role': 'excluded', 'reason': excluded}
        for gland in GLANDS:
            truth[gland] = largest_component(truth[gland])
        for organ in sorted(truth, key=lambda o: (o in GLANDS, labels[o])):  # adrenals last: they win overlaps
            label[truth[organ]] = labels[organ]

        header = reference.header.copy()
        header.set_data_dtype(np.uint8)
        nib.save(nib.Nifti1Image(label, ct.get_best_affine(), header), label_out)
        image_out.write_bytes(image_bytes)
        return {'name': name, 'source': source, 'case': case, 'role': role, 'reason': '',
                'classes_present': int(len(np.unique(label)) - 1)}
    except Exception as error:  # reported at the end; one bad file must not stop 1,500 cases
        for path in (image_out, label_out):
            path.unlink(missing_ok=True)
        return {'name': name, 'source': source, 'case': case, 'role': 'failed', 'reason': repr(error)}


def build(args):
    raw = args.data / 'nnUNet_raw' / DATASET
    for folder in ('imagesTr', 'labelsTr', 'imagesTs', 'labelsTs'):
        (raw / folder).mkdir(parents=True, exist_ok=True)
    labels = json.loads((fetch_d997(args.data) / 'dataset.json').read_text())['labels']
    folders = {'ts': args.data / 'sources/Totalsegmentator_dataset_v201.zip', 'amos': args.data / 'amos',
               'btcv': args.data / 'btcv', 'flare': args.data / 'flare'}
    scanned = list(csv.DictReader(open(args.scan)))
    # Known from the scan alone; the edge rule for small glands needs the masks and is applied per case.
    candidates = [r for r in scanned if not r['error'] and r['sides'] == 'ok' and float(r['slice_thickness_mm']) <= 5
                  and 'exclude_fragmented' not in (r['adrenal_left'], r['adrenal_right'])]
    development, test = five_folds([r for r in candidates if r['source'] != 'ts'])[0]
    development += [r for r in candidates if r['source'] == 'ts']
    held_out = {case_name(r) for r in test}
    log.info('%d scanned, %d candidates, %d in the AMOS/BTCV/FLARE test fifth', len(scanned), len(candidates), len(test))

    with Pool(args.workers) as pool:
        jobs = [(r, 'test' if case_name(r) in held_out else 'train', folders, args.pseudo, labels, raw) for r in candidates]
        rows = sorted(pool.imap_unordered(build_case, jobs, chunksize=2), key=lambda r: r['name'])
    write_csv(raw / 'build_dataset.csv', rows)
    for row in rows:
        if row['role'] in ('excluded', 'failed'):
            log.info('%s %s: %s', row['role'], row['name'], row['reason'])
    written = {r['name'] for r in rows if r['role'] == 'train'}
    (raw / 'dataset.json').write_text(json.dumps({
        'channel_names': {'0': 'CT'}, 'labels': labels, 'file_ending': '.nii.gz', 'numTraining': len(written),
        'overwrite_image_reader_writer': 'NibabelIOWithReorient'}, indent=2))  # D997's corrected (reorienting) reader
    splits = [{'train': [n for n in map(case_name, most) if n in written], 'val': [n for n in map(case_name, fifth) if n in written]}
              for most, fifth in five_folds(development)]
    preprocessed = args.data / 'nnUNet_preprocessed' / DATASET
    preprocessed.mkdir(parents=True, exist_ok=True)
    (preprocessed / 'splits_final.json').write_text(json.dumps(splits, indent=2))
    counts = {f"{s} {role}": sum(r['source'] == s and r['role'] == role for r in rows)
              for s in folders for role in ('train', 'test', 'excluded', 'failed')}
    log.info('fold 0: %d train, %d validation; %s', len(splits[0]['train']), len(splits[0]['val']), counts)
    if any(r['role'] == 'failed' for r in rows):
        raise SystemExit('Some cases failed; see the log. Nothing is marked finished.')
    (raw / 'build_dataset.json').write_text(json.dumps({  # written last: marks the build as finished
        'scan_csv_sha256': sha256(args.scan), 'pseudo_labels': str(args.pseudo), 'd997': D997, 'counts': counts,
        'build_dataset_csv_sha256': sha256(raw / 'build_dataset.csv')}, indent=2))


def preprocess(args):
    """preprocess.py's steps with D997's plans instead of AtlasNet's (editing preprocess.py would change the Dataset902
    cache recipe): only its plans and weights steps are swapped."""
    import preprocess as p
    folder = fetch_d997(args.data)

    def write_plans(root, preprocessed):
        from nnunetv2.experiment_planning.plans_for_pretraining.move_plans_between_datasets import move_plans_between_datasets
        source = preprocessed.parent / D997['source_dataset']
        source.mkdir(parents=True, exist_ok=True)
        shutil.copy(folder / 'plans.json', source / f"{D997['plans']}.json")
        move_plans_between_datasets(D997['source_dataset'], DATASET, D997['plans'], PLANS)  # reader from our dataset.json
        shutil.rmtree(source)

    p.DATASET, p.PLANS, p.write_plans, p.fetch_atlasnet_checkpoint = DATASET, PLANS, write_plans, lambda data: None
    sys.argv = ['preprocess.py', '--data', str(args.data), '--workers', str(args.workers)]
    p.main()
    (args.data / 'nnUNet_preprocessed' / DATASET / '.done').touch()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('step', choices=['build', 'preprocess'])
    parser.add_argument('--data', required=True, type=Path)
    parser.add_argument('--scan', type=Path, help="cohort_scan.py's results/cohort_scan.csv (build)")
    parser.add_argument('--pseudo', type=Path, help='pseudo-label prefix folder with amos/, btcv/, flare/ (build)')
    parser.add_argument('--workers', type=int, default=min(24, os.cpu_count()))
    args = parser.parse_args()
    args.data = args.data.resolve()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    build(args) if args.step == 'build' else preprocess(args)


def self_check():
    gland = np.zeros((20, 20, 20), bool)
    gland[5:8, 5:8, 5:8] = True  # 27 voxels at 1 mm: 0.027 mL, fully in view
    assert training_exclusion({'a': gland}, 0.001).endswith('exclude_small')
    gland[5:8, 5:8, 0:8] = True  # small but cut by a face: a partial gland, kept
    assert training_exclusion({'a': gland}, 0.001) == ''
    big = np.zeros((20, 20, 20), bool)
    big[0:12, 2:12, 2:12] = True  # 1.2 mL cut by a face (truncated): kept
    assert training_exclusion({'a': big}, 0.001) == ''
    big[15:19, 15:19, 15:19] = True  # 0.064 mL second piece is a speckle: kept
    assert training_exclusion({'a': big}, 0.001) == ''
    big[14:19, 14:19, 12:19] = True  # 0.175 mL second piece still a speckle (< 0.3 mL)
    assert training_exclusion({'a': big}, 0.001) == ''
    big[10:19, 14:19, 12:19] = True  # grows to 0.315 mL: fragmented
    assert training_exclusion({'a': big}, 0.001).endswith('exclude_fragmented')


if __name__ == '__main__':
    self_check()
    main()
