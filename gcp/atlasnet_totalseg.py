"""Existing-VM inference of the finished AtlasNet fine-tune on all TS scans, four replicas."""
import argparse
import base64
import hashlib
import json
import os
import shutil
from pathlib import Path
import subprocess
import sys
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor

ROOT = Path('/opt/atlasnet-totalseg')
MODEL = Path('/opt/adrenal-multiorgan/data/nnUNet_results/Dataset902_AdrenalMultiorgan/AdrenalMultiorgan__AtlasNetPlans__3d_fullres')
PREFIX = 'comparison-totalseg/ours-epoch1957'
REPO = 'ahigazy1/adrenal-multiorgan-model'
os.environ.update(OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', SITK_THREADS='2', nnUNet_compile='false')
sys.path.insert(0, '/opt/flare-run')


def sha(path):
    with open(path, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def auth():
    def get(url, headers):
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60) as r:
            return json.load(r)
    token = get('http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token',
                {'Metadata-Flavor': 'Google'})['access_token']
    value = get('https://secretmanager.googleapis.com/v1/projects/adrenal-seg/secrets/HF_TOKEN/versions/latest:access',
                {'Authorization': 'Bearer ' + token})
    os.environ['HF_TOKEN'] = base64.b64decode(value['payload']['data']).decode().strip()


def worker(index, smoke):
    import torch
    import nibabel as nib
    import numpy as np
    from huggingface_hub import HfApi
    from compare import load_predictor
    torch.set_num_threads(2)
    cases = json.loads((ROOT / 'cases.json').read_text())[index::4]
    if smoke:
        cases = cases[:1]
    output = ROOT / f'worker-{index}' / 'native_labels'
    output.mkdir(parents=True, exist_ok=True)
    predictor = load_predictor('ours-epoch1957', MODEL, 0, 'checkpoint_best.pth')
    api = HfApi()
    for start in range(0, len(cases), 8):
        chunk = cases[start:start + 8]
        pending = []
        for case in chunk:
            mask, record = output / f'{case}.nii.gz', output / f'{case}.json'
            if not (mask.exists() and record.exists() and json.loads(record.read_text())['sha256'] == sha(mask)):
                pending.append(case)
        began = time.time()
        if pending:
            predictor.predict_from_files([[str(ROOT / 'imagesTs' / f'{c}_0000.nii.gz')] for c in pending],
                                         [str(output / c) for c in pending], save_probabilities=False,
                                         overwrite=True, num_processes_preprocessing=3,
                                         num_processes_segmentation_export=2)
        for case in chunk:
            path = output / f'{case}.nii.gz'
            img = nib.load(path); ref = nib.load(ROOT / 'imagesTs' / f'{case}_0000.nii.gz')
            a = np.asanyarray(img.dataobj)
            assert img.shape == ref.shape and np.allclose(img.affine, ref.affine, atol=1e-3)
            assert np.isfinite(a).all() and np.equal(a, np.floor(a)).all() and a.min() >= 0 and a.max() <= 9
            voxel = abs(np.linalg.det(img.affine[:3, :3])) / 1000
            record = {'case': case, 'sha256': sha(path), 'bytes': path.stat().st_size,
                      'volumes_ml': {str(k): int((a == k).sum()) * voxel for k in range(1, 10)}}
            (output / f'{case}.json').write_text(json.dumps(record))
        api.upload_folder(repo_id=REPO, folder_path=output, path_in_repo=f'{PREFIX}/native_labels',
                          allow_patterns=[f'{c}.*' for c in chunk], commit_message=f'AtlasNet TS worker {index}: {start + len(chunk)}/{len(cases)}')
        print(f'WORKER {index}: {start + len(chunk)}/{len(cases)} verified/uploaded, chunk {time.time()-began:.1f}s', flush=True)


def main():
    auth()
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', type=int)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    ROOT.mkdir(exist_ok=True)
    if args.worker is not None:
        worker(args.worker, args.smoke)
        return
    import torch
    from huggingface_hub import HfApi, hf_hub_download
    checkpoint = MODEL / 'fold_0/checkpoint_best.pth'
    evidence = hf_hub_download(REPO, 'comparison/ours-epoch1957/predict.json', revision='50cc899ffc06d4ac401ee6f2c9bc9230d1a16495')
    expected = json.loads(Path(evidence).read_text())['checkpoint_sha256']
    assert sha(checkpoint) == expected, 'Checkpoint differs from evaluated epoch1957'
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    print('Checkpoint verified', expected, 'saved epoch', saved.get('current_epoch'), flush=True)
    del saved
    archive = '/opt/adrenal-multiorgan/data/sources/Totalsegmentator_dataset_v201.zip'
    with zipfile.ZipFile(archive) as z:
        members = sorted(n for n in z.namelist() if n.endswith('/ct.nii.gz'))
        required = sum(z.getinfo(n).file_size for n in members)
    assert len(members) == 1228
    assert shutil.disk_usage(ROOT).free > required + 20 * 1024**3, 'Insufficient extraction/output space'
    images = ROOT / 'imagesTs'; images.mkdir(exist_ok=True)
    def extract(member):
        case = 'ts_' + member.split('/')[-2]
        dest = images / f'{case}_0000.nii.gz'
        if not dest.exists():
            with zipfile.ZipFile(archive) as z:
                contents = z.read(member)
            temp = dest.with_suffix('.tmp'); temp.write_bytes(contents); temp.replace(dest)
        return case
    with ThreadPoolExecutor(12) as pool:
        cases = list(pool.map(extract, members))
    (ROOT / 'cases.json').write_text(json.dumps(cases, indent=2))
    print('Prepared', len(cases), 'CT scans', flush=True)
    import compare
    compare.RAW = ROOT
    torch.set_num_threads(4)
    gate = compare.check_batching('ours-epoch1957', MODEL, 0, 'checkpoint_best.pth')
    (ROOT / 'batching_check.json').write_text(json.dumps(gate, indent=2))
    assert gate['passed'], gate
    torch.cuda.empty_cache()
    def launch(smoke):
        processes = []
        for i in range(4):
            log = open(ROOT / f'worker-{i}.log', 'a')
            cmd = [sys.executable, '-u', __file__, '--worker', str(i)] + (['--smoke'] if smoke else [])
            processes.append(subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT))
        codes = [p.wait() for p in processes]
        assert codes == [0] * 4, codes
    launch(True)
    print('Four-replica native-grid/upload smoke passed', flush=True)
    record = {'checkpoint_sha256': expected, 'cases': 1228, 'replicas': 4, 'preprocessors_per_replica': 3,
              'exporters_per_replica': 2, 'threads_per_worker': 2, 'mirroring': False, 'dataset': 'all TotalSegmentator v2.0.1',
              'note': 'Includes scans used in fine-tuning; not a held-out performance estimate'}
    (ROOT / 'run.json').write_text(json.dumps(record, indent=2))
    api = HfApi()
    for name in ('run.json', 'cases.json', 'batching_check.json'):
        api.upload_file(repo_id=REPO, path_or_fileobj=ROOT / name, path_in_repo=f'{PREFIX}/{name}')
    launch(False)
    records = [json.loads((ROOT / f'worker-{i % 4}' / 'native_labels' / f'{case}.json').read_text())
               for i, case in enumerate(cases)]
    assert len(records) == 1228 and {r['case'] for r in records} == set(cases)
    remote = {p.path: p for p in api.list_repo_tree(REPO, path_in_repo=f'{PREFIX}/native_labels', recursive=True) if hasattr(p, 'size')}
    for r in records:
        f = remote[f'{PREFIX}/native_labels/{r["case"]}.nii.gz']
        assert f.size == r['bytes']
        if f.lfs:
            assert f.lfs.sha256 == r['sha256']
        else:
            path = next(ROOT.glob(f'worker-*/native_labels/{r["case"]}.nii.gz'))
            content = path.read_bytes()
            assert f.blob_id == hashlib.sha1(f'blob {len(content)}\0'.encode() + content).hexdigest()
    completion = json.dumps({'run': record, 'outputs': records}, indent=2)
    api.upload_file(repo_id=REPO, path_or_fileobj=completion.encode(), path_in_repo=f'{PREFIX}/complete.json')
    (ROOT / 'complete.json').write_text(completion)
    print('COMPLETE: 1228 masks verified and uploaded', flush=True)


if __name__ == '__main__':
    main()
