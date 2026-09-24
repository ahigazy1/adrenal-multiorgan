"""TotalSegmentator pseudo-labels for AMOS22 CT, BTCV, FLARE22 and RAOS, in D998's 66 classes.

    python pseudolabel.py --data /opt/pseudo --workers 4

One uint8 NIfTI per labelled scan, on the scan's own grid, with D998's ids (dataset.json below); every other
TotalSegmentator class is 0, kidney cysts are merged into their kidney. Nothing else is changed: merging with the real
labels (real labels win, pseudo-labels become the ignore label) is a later step.

Files go to the private dataset ahigazy1/adrenal-multiorgan-cache under pseudolabels/totalseg-<version>/<source>/<case>.nii.gz,
uploaded every UPLOAD_EVERY scans. Cases already there are skipped, so an interrupted run continues where it stopped.
"""
import argparse
import json
import multiprocessing
import platform
import time
from pathlib import Path

import nibabel as nib
import numpy as np
from huggingface_hub import HfApi, snapshot_download

REPO = 'ahigazy1/adrenal-multiorgan-cache'
UPLOAD_EVERY = 25
# Sources at pinned revisions (the same as ../prepare.sh), downloaded one after the other (parallel gets HTTP 429).
SOURCES = {
    'amos': ('MedOtter/amos22-ct-dataset', 'c67f7c01e66277038d87975b03b73ece489a3035', ['train/*', 'valid/*']),
    'btcv': ('lingheng123/btcv', 'c1728b451a00c054875a0a97d7658d1eaf8362b5', ['RawData/Training/*']),
    'flare': ('MedOtter/FLARE22', 'ab0b99b53e2183fe59321b5888867c6c7cb0792a', ['images/*', 'labels/*']),
    'raos': ('ahigazy1/AdrenalSeg-Sources', None, ['RAOS-Real/*/images*/*', 'RAOS-Real/*/labels*/*']),
}
# D998 (Dataset998_TotalSeg66classes) label ids; the names are TotalSegmentator v2's own.
D998 = ['spleen', 'kidney_right', 'kidney_left', 'gallbladder', 'liver', 'stomach', 'pancreas', 'adrenal_gland_right',
        'adrenal_gland_left', 'lung_upper_lobe_left', 'lung_lower_lobe_left', 'lung_upper_lobe_right',
        'lung_middle_lobe_right', 'lung_lower_lobe_right', 'esophagus', 'trachea', 'thyroid_gland', 'small_bowel',
        'duodenum', 'colon', 'urinary_bladder', 'prostate', 'sacrum', 'vertebrae_S1',
        *[f'vertebrae_{v}' for v in ['L5', 'L4', 'L3', 'L2', 'L1', 'T12', 'T11', 'T10', 'T9', 'T8', 'T7', 'T6', 'T5',
                                     'T4', 'T3', 'T2', 'T1', 'C7', 'C6', 'C5', 'C4', 'C3', 'C2', 'C1']],
        'heart', 'aorta', 'pulmonary_vein', 'brachiocephalic_trunk', 'subclavian_artery_right',
        'subclavian_artery_left', 'common_carotid_artery_right', 'common_carotid_artery_left',
        'brachiocephalic_vein_left', 'brachiocephalic_vein_right', 'atrial_appendage_left', 'superior_vena_cava',
        'inferior_vena_cava', 'portal_vein_and_splenic_vein', 'iliac_artery_left', 'iliac_artery_right',
        'iliac_vena_left', 'iliac_vena_right']
MERGED = {'kidney_cyst_left': 'kidney_left', 'kidney_cyst_right': 'kidney_right'}


def lookup():
    """TotalSegmentator 'total' id -> D998 id."""
    from totalsegmentator.map_to_binary import class_map
    ids = {name: i for i, name in class_map['total'].items()}
    missing = [n for n in [*D998, *MERGED] if n not in ids]
    if missing:
        raise SystemExit(f'TotalSegmentator has no class {missing}: the class names changed')
    lut = np.zeros(max(ids.values()) + 1, np.uint8)
    for d998_id, name in enumerate(D998, 1):
        lut[ids[name]] = d998_id
    for cyst, kidney in MERGED.items():
        lut[ids[cyst]] = D998.index(kidney) + 1
    return lut


def scans(data):
    """(source, case, image) for every labelled scan; case is the label file's name without .nii.gz."""
    for source, (repo, revision, patterns) in SOURCES.items():
        root = Path(snapshot_download(repo, repo_type='dataset', revision=revision, allow_patterns=patterns,
                                      local_dir=data / 'sources' / source))
        if source == 'amos':
            for labels, images in (('train/labelsTr', 'train/imagesTr'), ('valid/labelsVa', 'valid/imagesVa')):
                for label in sorted((root / labels).glob('*.nii.gz')):
                    yield source, label.name[:-7], root / images / label.name
        elif source == 'btcv':
            for label in sorted((root / 'RawData/Training/label').glob('*.nii.gz')):
                yield source, label.name[:-7], root / 'RawData/Training/img' / label.name.replace('label', 'img')
        elif source == 'flare':
            for label in sorted((root / 'labels').glob('*.nii.gz')):
                yield source, label.name[:-7], root / 'images' / f'{label.name[:-7]}_0000.nii.gz'
        else:  # RAOS: three sets (cancer, after surgery, organs removed), Tr and Ts image naming differ
            for label in sorted(root.glob('RAOS-Real/*/labels*/*.nii.gz')):
                group = 'raos' + label.parent.parent.name.split('(Set')[1][0]
                folder = label.parent.parent / label.parent.name.replace('labels', 'images')
                image = next(folder / f for f in (f'{label.name[:-7]}_0000.nii.gz', label.name) if (folder / f).exists())
                yield source, f"{group}_{label.name[:-7].replace('.', '-')}", image


def label_one(job):
    try:
        return label_scan(*job)
    except Exception as error:  # one unreadable scan must not stop the others; it is listed in the manifest
        return f'ERROR {job[0]}: {error}'


def label_scan(image, target, lut):
    from totalsegmentator.python_api import totalsegmentator
    ct = nib.load(image)
    seg = totalsegmentator(ct, None, ml=True, task='total', device='gpu', quiet=True, nr_thr_resamp=4, nr_thr_saving=1)
    values = np.asanyarray(seg.dataobj)
    if values.shape != ct.shape[:3] or not np.allclose(seg.affine, ct.affine, atol=1e-3):
        raise RuntimeError(f'{image}: TotalSegmentator output is not on the scan grid')
    out = nib.Nifti1Image(lut[values.astype(np.int64)], ct.affine)
    out.set_data_dtype(np.uint8)
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + '.partial.nii.gz')
    nib.save(out, partial)
    partial.rename(target)
    return target


def upload(api, folder):
    api.upload_large_folder(REPO, folder, repo_type='dataset', allow_patterns=['pseudolabels/*'],
                            ignore_patterns=['*.partial.nii.gz'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4, help='TotalSegmentator processes sharing the GPU')
    args = parser.parse_args()
    import torch
    from importlib.metadata import version
    prefix = f"pseudolabels/totalseg-{version('TotalSegmentator')}"
    upload_root = args.data / 'upload'
    api = HfApi()
    done = {f for f in api.list_repo_files(REPO, repo_type='dataset') if f.startswith(prefix) and f.endswith('.nii.gz')}
    lut = lookup()
    jobs = []
    for source, case, image in scans(args.data):
        target = upload_root / prefix / source / f'{case}.nii.gz'
        if target.relative_to(upload_root).as_posix() not in done and not target.exists():
            jobs.append((image, target, lut))
    total = len(jobs) + len(done)
    print(f'{len(done)} already on Hugging Face, {len(jobs)} to label', flush=True)
    start, finished, failed = time.time(), 0, []
    with multiprocessing.get_context('spawn').Pool(args.workers) as pool:
        for result in pool.imap_unordered(label_one, jobs):
            finished += 1
            if isinstance(result, str):
                failed.append(result)
                print(result, flush=True)
            if finished % 5 == 0 or finished == len(jobs):
                rate = (time.time() - start) / finished
                print(f'{finished}/{len(jobs)} cases written, {rate:.0f} s per case, '
                      f'{rate * (len(jobs) - finished) / 3600:.1f} h left', flush=True)
            if finished % UPLOAD_EVERY == 0:
                upload(api, upload_root)
    labelled = sorted(p.relative_to(upload_root / prefix).as_posix() for p in (upload_root / prefix).rglob('*.nii.gz'))
    manifest = {'labels': {'background': 0, **{n: i for i, n in enumerate(D998, 1)}}, 'merged_into_kidney': MERGED,
                'totalsegmentator': version('TotalSegmentator'), 'nnunetv2': version('nnunetv2'), 'torch': torch.__version__,
                'gpu': torch.cuda.get_device_name(0), 'python': platform.python_version(), 'task': 'total', 'fast': False,
                'sources': {s: {'repo': r, 'revision': rev} for s, (r, rev, _) in SOURCES.items()},
                'scans_labelled_on_this_machine': labelled, 'failed_on_this_machine': failed, 'scans_total': total}
    (upload_root / prefix / 'dataset.json').write_text(json.dumps(manifest, indent=1))
    upload(api, upload_root)
    print(f'=== pseudo-labels complete: https://huggingface.co/datasets/{REPO}/tree/main/{prefix}', flush=True)


if __name__ == '__main__':
    main()
