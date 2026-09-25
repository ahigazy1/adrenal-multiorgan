"""TotalSegmentator pseudo-labels for labelled AMOS22 CT, BTCV and FLARE22.

    python pseudolabels/pseudolabel.py --data /opt/pseudo

One uint8 NIfTI per labelled scan, on the scan's own grid, with D997/D998's shared ids; every other
TotalSegmentator class is 0, kidney cysts are merged into their kidney. Nothing else is changed: merging with the real
labels (real labels win, pseudo-labels become the ignore label) is a later step.

Files go to the private dataset ahigazy1/adrenal-multiorgan-cache under
pseudolabels/totalseg-<version>/<recipe>/<source>/<case>.nii.gz. The first scan, then batches of 25,
are committed with a checksum manifest. Only verified remote results are skipped on resume.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
import os
import sys
import time
from pathlib import Path, PurePosixPath

import nibabel as nib
import numpy as np
from huggingface_hub import snapshot_download

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hf_cache import Hub, digest, encoded, file_info, safe_path, verify_remote
from runtime import THREADS, setup as setup_runtime

REPO = 'ahigazy1/adrenal-pseudolabels'
UPLOAD_EVERY = 25
NNUNET_COMMIT = '940dcd3ba8e9f9a1c21885ca52acd2f06328b6fc'
# Sources at pinned revisions (the same as ../prepare.sh), downloaded one after the other (parallel gets HTTP 429).
SOURCES = {
    'amos': ('MedOtter/amos22-ct-dataset', 'c67f7c01e66277038d87975b03b73ece489a3035', 300),
    'btcv': ('lingheng123/btcv', 'c1728b451a00c054875a0a97d7658d1eaf8362b5', 30),
    'flare': ('MedOtter/FLARE22', 'ab0b99b53e2183fe59321b5888867c6c7cb0792a', 50),
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


def scans(api):
    """Validate the pinned inventory before downloading images or running inference."""
    cases = {}
    for source, (repo, revision, expected) in SOURCES.items():
        names = set(api.list_repo_files(repo, repo_type='dataset', revision=revision))
        count = 0
        for name in sorted(names):
            label = PurePosixPath(name)
            if not name.endswith('.nii.gz'):
                continue
            if source == 'amos' and label.parent.as_posix() in ('train/labelsTr', 'valid/labelsVa'):
                candidates = [name.replace('/labels', '/images')]
            elif source == 'btcv' and label.parent.as_posix() == 'RawData/Training/label':
                candidates = [name.replace('/label/label', '/img/img')]
            elif source == 'flare' and label.parent.as_posix() == 'labels':
                candidates = [f'images/{label.name[:-7]}_0000.nii.gz']
            else:
                continue
            matches = [image for image in candidates if image in names]
            if len(matches) != 1:
                raise RuntimeError(f'{source}: expected one image for {name}, found {matches}')
            case = label.name[:-7]
            key = f'{source}/{case}.nii.gz'
            if key in cases:
                raise RuntimeError(f'Duplicate output: {key}')
            cases[key] = (source, matches[0])
            count += 1
        if count != expected:
            raise RuntimeError(f'{source}: expected {expected} labelled scans, found {count}')
    return cases


def label_scan(job):
    image, target, lut = job
    started = time.monotonic()
    print(f'=== case start: {target.parent.name}/{target.name}; worker={os.getpid()}', flush=True)
    import torch
    from totalsegmentator.python_api import totalsegmentator
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; refusing CPU fallback')
    ct = nib.load(image)
    if len(ct.shape) != 3:
        raise RuntimeError(f'{image}: expected a 3D CT')
    seg = totalsegmentator(ct, None, ml=True, task='total', device='gpu', quiet=True,
                          nr_thr_resamp=THREADS, nr_thr_saving=1, resampling_order=1)
    values = np.asanyarray(seg.dataobj)
    if values.shape != ct.shape[:3] or not np.allclose(seg.affine, ct.affine, rtol=0, atol=1e-3):
        raise RuntimeError(f'{image}: TotalSegmentator output is not on the scan grid')
    if values.dtype.kind not in 'iu' or values.min() < 0 or values.max() >= len(lut):
        raise RuntimeError(f'{image}: invalid TotalSegmentator label ids')
    out = nib.Nifti1Image(lut[values], ct.affine, ct.header.copy())
    out.set_qform(ct.get_qform(), int(ct.header['qform_code']))
    out.set_sform(ct.get_sform(), int(ct.header['sform_code']))
    out.set_data_dtype(np.uint8)
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + '.partial.nii.gz')
    nib.save(out, partial)
    partial.replace(target)
    print(f'=== case done: {target.parent.name}/{target.name}; {time.monotonic() - started:.1f}s; upload pending', flush=True)
    return target


def worker_setup(data, gate):
    from totalsegmentator.config import get_weights_dir, setup_totalseg, set_config_key
    os.environ['TOTALSEG_WEIGHTS_PATH'] = str(get_weights_dir())
    # TotalSegmentator updates config.json per scan. Give concurrent workers separate configs.
    os.environ['TOTALSEG_HOME_DIR'] = str(data / 'worker-config' / str(os.getpid()))
    setup_totalseg()
    set_config_key('send_usage_stats', False)
    set_config_key('statistics_disclaimer_shown', True)
    setup_runtime(gate)


def recipe():
    from importlib.metadata import distribution, version
    from totalsegmentator.config import get_weights_dir
    from totalsegmentator.map_to_binary import class_map
    weights = get_weights_dir()
    checkpoints = sorted(weights.rglob('*.pth'))
    if not checkpoints:
        raise RuntimeError('Download TotalSegmentator weights first')
    direct = json.loads(distribution('nnunetv2').read_text('direct_url.json') or '{}')
    if direct.get('vcs_info', {}).get('commit_id') != NNUNET_COMMIT:
        raise RuntimeError('nnU-Net is not at the pinned GitHub commit; use startup.sh')
    return {'sources': SOURCES, 'labels': D998,
            'label_ids': {'background': 0, **{name: i for i, name in enumerate(D998, 1)}},
            'source_label_ids': class_map['total'],
            'merged': MERGED, 'task': 'total', 'fast': False,
            'resampling': 'SimpleITK array zoom; endpoint-aligned; CT linear; labels nearest',
            'orientation': 'TotalSegmentator canonicalization and inverse; native input grid',
            'cpu_threads_per_worker': THREADS, 'gpu_inference_concurrency': 1,
            'runtime_code': file_info(Path(__file__).with_name('runtime.py'))['sha256'],
            'nnunet_commit': NNUNET_COMMIT,
            'versions': {n: version(n) for n in ('TotalSegmentator', 'nnunetv2', 'torch', 'numpy', 'scipy', 'SimpleITK', 'scikit-image')},
            'code': file_info(Path(__file__))['sha256'],
            'requirements': file_info(Path(__file__).with_name('requirements.txt'))['sha256'],
            'weights': {p.relative_to(weights).as_posix(): file_info(p)['sha256']
                        for p in checkpoints + sorted(weights.rglob('plans.json'))}}


def run(data, workers, hub, spec):
    cases = scans(hub.api)
    prefix = f"pseudolabels/totalseg-{spec['versions']['TotalSegmentator']}/{digest(spec)[:16]}"
    manifest_path = f'{prefix}/dataset.json'
    revision, entries = hub.info()
    manifest = {'recipe': spec, 'files': {}, 'complete': False, 'scans_total': len(cases)}
    if manifest_path in entries:
        manifest = json.loads(hub.get(manifest_path, revision, data / 'manifests').read_text())
        if digest(manifest['recipe']) != digest(spec) or not set(manifest['files']) <= cases.keys():
            raise RuntimeError('Incompatible pseudo-label manifest')
        verify_remote(entries, prefix, manifest)
    pending = [name for name in cases if name not in manifest['files']]
    print(f"{len(manifest['files'])}/{len(cases)} verified on Hugging Face; {len(pending)} pending", flush=True)
    lut = lookup()
    start = time.monotonic()
    completed = 0
    # First case checks the real GPU -> NIfTI -> HF path before spending on a batch.
    batches = [pending[:1]] + [pending[i:i + UPLOAD_EVERY] for i in range(1, len(pending), UPLOAD_EVERY)]
    context = multiprocessing.get_context('spawn')
    gate = context.BoundedSemaphore(1)
    # Reuse workers across uploads instead of importing the teacher 12 times per batch.
    with ProcessPoolExecutor(max_workers=workers, mp_context=context,
                             initializer=worker_setup, initargs=(data, gate)) as pool:
        for batch in batches:
            if not batch:
                continue
            roots = {}
            for source in sorted({cases[name][0] for name in batch}):
                repo, source_revision, _ = SOURCES[source]
                roots[source] = Path(snapshot_download(repo, repo_type='dataset', revision=source_revision,
                    allow_patterns=[cases[name][1] for name in batch if cases[name][0] == source],
                    local_dir=data / 'sources' / source, max_workers=4))
            jobs = [(safe_path(roots[cases[name][0]], cases[name][1]), safe_path(data / prefix, name), lut) for name in batch]
            # Executor workers are non-daemon: nnU-Net can create preprocessing/export processes.
            if workers == 1 or len(batch) == 1:
                outputs = [label_scan(job) for job in jobs]
            else:
                outputs = list(pool.map(label_scan, jobs))
            manifest['files'].update({name: file_info(path) for name, path in zip(batch, outputs)})
            manifest['complete'] = set(manifest['files']) == cases.keys()
            additions = {f'{prefix}/{name}': path for name, path in zip(batch, outputs)}
            additions[manifest_path] = encoded(manifest)
            revision = hub.commit(additions, f"Pseudo-labels: {len(manifest['files'])}/{len(cases)}", revision)
            current_revision, entries = hub.info()
            verify_remote(entries, prefix, manifest)
            # Do not overwrite another writer's progress. Restart to reload its manifest.
            if current_revision != revision:
                raise RuntimeError('Repository changed during upload; restart to refresh the manifest')
            completed += len(batch)
            rate = (time.monotonic() - start) / completed
            print(f"Uploaded {len(manifest['files'])}/{len(cases)} cases; {rate:.1f} s/case; "
                  f"{rate * (len(pending) - completed) / 3600:.1f} h left", flush=True)
    if not manifest['complete'] or set(manifest['files']) != cases.keys():
        raise RuntimeError('Pseudo-label dataset is incomplete')
    print(f'=== pseudo-labels complete: https://huggingface.co/datasets/{REPO}/tree/main/{prefix}', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=12, help='Case workers, 4 CPU threads each; one GPU inference at a time')
    args = parser.parse_args()
    if not 1 <= args.workers <= 12:
        parser.error('--workers must be between 1 and 12')
    # Inherited by spawned workers before NumPy/BLAS or nnU-Net are imported.
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                'SITK_THREADS', 'ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS', 'nnUNet_def_n_proc'):
        os.environ[key] = str(THREADS)
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; refusing CPU fallback')
    print(f'GPU: {torch.cuda.get_device_name(0)}; workers: {args.workers}', flush=True)
    spec = recipe()
    worker_setup(args.data, multiprocessing.get_context('spawn').BoundedSemaphore(1))
    # Public pseudo-label output is explicitly authorized; later privatization is OK.
    run(args.data, args.workers, Hub(REPO, 'dataset', allow_public=True), spec)


if __name__ == '__main__':
    main()
