"""Compare models on the 279 held-out test scans (run on any CUDA machine, e.g. a Colab A100).

    python compare.py ours atlasnet labmate997 labmate998      # or a subset; finished scans are skipped

For every model: download it from Hugging Face, segment the test scans with nnU-Net's own predictor
(no mirroring, tile step 0.5, several patches per network call: batched_predictor.py), rename its labels to our nine organs, score with evaluate.py and upload
predictions and tables to the model repository under comparison/<model>/.

Resampling. Each model gets the CT resampled the way it was trained (its own plans). The network output of the
models whose plans use nnU-Net's default resampler is resampled with the SimpleITK function instead: same result
to float precision (PLAN.md), but multithreaded. AtlasNet keeps its own torch resampler; ours is SimpleITK already.

The other models were trained on TotalSegmentator scans with other splits, so some of our TotalSegmentator test
scans were training scans for them: read their numbers per source (evaluation/*_by_source.csv).
"""
import argparse
import json
import logging
import os
import psutil
import resource
import signal
import subprocess
import sys
import tarfile
import threading
import time
import zipfile
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download, snapshot_download

from evaluate import LABELS
from hf_cache import CACHE_REPO
from preprocess import ATLASNET, DATASET, RESAMPLING
from train import DATA, MODEL_FOLDER, ROOT, sha256  # importing train also sets the nnU-Net folders and trainer path

MODEL_REPO = os.environ['ADRENAL_HF_REPO']
RAW = DATA / 'nnUNet_raw' / DATASET
D998 = 'Dataset998_TotalSeg66classes/nnUNetTrainer_2000epochs_NoMirroring__nnUNetResEncLPlans_16GB__3d_fullres_bs8'
# model -> Hugging Face repo, repo type, folder in the repo, fold, checkpoint
MODELS = {'ours': (MODEL_REPO, 'model', MODEL_FOLDER.name, 0, 'checkpoint_best.pth'),
          'labmate997': ('ahigazy1/adrenal-training-models', 'model', 'labmate', 0, 'checkpoint_final.pth'),
          'labmate998': ('ahigazy1/AdrenalSeg-Sources', 'dataset', D998, 0, 'checkpoint_final.pth'),
          'atlasnet': (ATLASNET['repo'], 'model', None, 'all', 'checkpoint_final.pth'),
          'labmate997-reoriented': ('ahigazy1/adrenal-training-models', 'model', 'labmate', 0, 'checkpoint_final.pth')}
# labmate997's dataset.json names no image reader, so nnU-Net's SimpleITK reader was used and the AMOS and BTCV scans
# (not stored RAS) came out mirrored left/right. The reorienting reader every other model uses fixes that.
READER = {'labmate997-reoriented': 'NibabelIOWithReorient'}
NAMES = {'adrenal_left': ['adrenal_left', 'adrenal_gland_left'], 'adrenal_right': ['adrenal_right', 'adrenal_gland_right'],
         'inferior_vena_cava': ['inferior_vena_cava', 'postcava']}  # other organs have the same name everywhere
MAX_CHANGED_FRACTION = 1e-4  # accepted: batching changed 1 to 9 voxels per million on the first check; nnU-Net itself repeats exactly
ABORT_CONDITION = f"labels_changed / voxels > {MAX_CHANGED_FRACTION} (batched against nnU-Net's own sliding window)"
CHUNK = 20  # scans predicted between two uploads
log = logging.getLogger('compare')


def fetch_test_scans():
    """The test scans are the last files of the preprocessed cache: read parts from the end until all are here."""
    api = HfApi()
    complete = next(f.path for f in api.list_repo_tree(CACHE_REPO, 'preprocessed', recursive=True, repo_type='dataset')
                    if f.path.endswith('complete.json'))
    record = json.loads(Path(hf_hub_download(CACHE_REPO, complete, repo_type='dataset')).read_text())
    wanted = {name for name in record['files'] if name.startswith('nnUNet_raw/')}
    for part in sorted(record['archives'], reverse=True):
        if all((DATA / name).exists() for name in wanted):
            break
        log.info('Downloading %s', part)
        path = hf_hub_download(CACHE_REPO, f"{Path(complete).parent.as_posix()}/{record['content_id']}/{part}", repo_type='dataset')
        with tarfile.open(path) as tar:
            tar.extractall(DATA, members=[m for m in tar if m.name in wanted], filter='data')
    for name in wanted:
        if sha256(DATA / name) != record['files'][name]['sha256']:
            raise SystemExit(f'{name} differs from the published cache')
    log.info('%d test scans verified', len(list((RAW / 'imagesTs').glob('*.nii.gz'))))


def fetch_model(name):
    repo, repo_type, folder, fold, checkpoint = MODELS[name]
    target = DATA / 'comparison_models' / name
    if name == 'atlasnet':
        archive = hf_hub_download(repo, ATLASNET['archive'], revision=ATLASNET['revision'])
        prefix = str(Path(ATLASNET['checkpoint']).parent.parent)
        with zipfile.ZipFile(archive) as z:
            for member in ('dataset.json', 'plans.json', f'fold_{fold}/{checkpoint}'):
                if not (target / member).exists():
                    (target / member).parent.mkdir(parents=True, exist_ok=True)
                    (target / member).write_bytes(z.read(next(n for n in z.namelist() if n.endswith(f'{prefix}/{member}'))))
        return target
    for member in ('dataset.json', 'plans.json', f'fold_{fold}/{checkpoint}'):
        hf_hub_download(repo, f'{folder}/{member}', repo_type=repo_type, local_dir=target)
    return target / folder


def label_mapping(model):
    labels = json.loads((model / 'dataset.json').read_text())['labels']
    mapping = {number: value for organ, value in LABELS.items() for label, number in labels.items()
               if label.startswith(tuple(NAMES.get(organ, [organ])))}  # AtlasNet: liver_segment_1..8, pancreas_head/body/tail
    if set(mapping.values()) != set(LABELS.values()):
        raise SystemExit(f'{model} does not segment all nine organs')
    return mapping


def load_predictor(name, model, fold, checkpoint):
    import torch
    import nnunetv2.inference.predict_from_raw_data as nnunet
    from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
    from batched_predictor import BatchedPredictor
    # ponytail: the other groups' trainer classes are not in this repository. Only the network is needed and their
    # trainers build the standard one from plans.json, so an unknown trainer name falls back to nnUNetTrainer.
    find = nnunet.recursive_find_trainer_class_by_name
    nnunet.recursive_find_trainer_class_by_name = lambda trainer: find(trainer) or nnUNetTrainer
    predictor = BatchedPredictor(tile_step_size=0.5, use_mirroring=False, perform_everything_on_device=True,
                                 device=torch.device('cuda'))
    predictor.initialize_from_trained_model_folder(str(model), use_folds=(fold,), checkpoint_name=checkpoint)
    if name in READER:
        predictor.dataset_json['overwrite_image_reader_writer'] = READER[name]
    configuration = predictor.configuration_manager.configuration
    if configuration['resampling_fn_probabilities'] == 'resample_data_or_seg_to_shape':
        configuration['resampling_fn_probabilities'] = RESAMPLING['resampling_fn_probabilities']
    return predictor


def check_batching(name, model, fold, checkpoint):
    """One scan through nnU-Net's own sliding window and through the batched one: the network outputs must agree."""
    import torch
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    predictor = load_predictor(name, model, fold, checkpoint)
    scans = sorted((RAW / 'imagesTs').glob('*_0000.nii.gz'), key=lambda path: path.stat().st_size)
    scan = scans[len(scans) // 2]  # the scan of median file size
    data = predictor.configuration_manager.preprocessor_class(verbose=False).run_case(
        [str(scan)], None, predictor.plans_manager, predictor.configuration_manager, predictor.dataset_json)[0]
    batched = type(predictor)

    def logits_of(kind):
        predictor.__class__ = kind
        return predictor.predict_logits_from_preprocessed_data(torch.from_numpy(data)).float()

    stock = logits_of(nnUNetPredictor)
    labels = stock.argmax(0)
    result = {'model': name, 'scan': scan.name, 'shape': list(data.shape), 'voxels': labels.numel()}
    # nnU-Net's own predictor a second time: how much it differs from itself (cuDNN autotuning, torch.compile)
    for key, kind in (('stock_repeated', nnUNetPredictor), ('batched', batched)):
        other = logits_of(kind)
        result[key] = {'max_delta_logit': (other - stock).abs().max().item(),
                       'labels_changed': int((other.argmax(0) != labels).sum())}
        del other
    result['max_delta_logit'], result['labels_changed'] = result['batched'].values()
    result['abort_condition'] = ABORT_CONDITION
    result['changed_fraction'] = result['labels_changed'] / result['voxels']
    result['passed'] = result['changed_fraction'] <= MAX_CHANGED_FRACTION
    log.info('Batching check %s', result)
    return result


def predict(name, model, output, fold, checkpoint):
    import nibabel as nib
    import numpy as np
    predictor = load_predictor(name, model, fold, checkpoint)
    configuration = predictor.configuration_manager.configuration
    mapping = label_mapping(model)
    record = {'model': name, 'checkpoint_sha256': sha256(model / f'fold_{fold}' / checkpoint), 'mirroring': False,
              'tile_step_size': 0.5, 'predictor': 'batched_predictor.py',
              'image_reader': predictor.dataset_json.get('overwrite_image_reader_writer', 'SimpleITKIO'), 'resampling_fn_data': configuration['resampling_fn_data'],
              'resampling_fn_probabilities': configuration['resampling_fn_probabilities'],
              'label_mapping': {str(k): v for k, v in mapping.items()}}
    record_path = output / 'predict.json'
    if record_path.exists() and json.loads(record_path.read_text()) != record:
        raise SystemExit(f'{output} holds predictions of a different model; move it away first')
    native = output / 'native_labels'
    native.mkdir(parents=True, exist_ok=True)
    record_path.write_text(json.dumps(record, indent=2))
    # scans predicted by an earlier, lost machine: fetch them so they are not predicted again
    snapshot_download(MODEL_REPO, allow_patterns=[f'comparison/{name}/native_labels/*'], local_dir=DATA)
    api = HfApi()
    # Export workers hold the full-size network output of a scan (classes x voxels, float32, twice), so models with many
    # classes get fewer of them. The workers are new processes and read SITK_THREADS when they start.
    # A whole-body scan at 1 mm is ~250 M voxels: 10 GB per class-volume copy for our model, 65 GB for the 66-class ones.
    # A machine that runs out of memory simply vanishes (twice so far), so one export at a time and the threads go to it.
    cpus = os.cpu_count()
    exporters = 1
    os.environ['SITK_THREADS'] = str(cpus)
    scans = sorted((RAW / 'imagesTs').glob('*_0000.nii.gz'))
    for start in range(0, len(scans), CHUNK):  # after every chunk the predictions are safe on Hugging Face
        chunk = [s for s in scans[start:start + CHUNK] if not (native / s.name.replace('_0000', '')).exists()]
        if not chunk:
            continue
        predictor.predict_from_files([[str(s)] for s in chunk], [str(native / s.name[:-12]) for s in chunk],
                                     save_probabilities=False, overwrite=False,
                                     num_processes_preprocessing=2, num_processes_segmentation_export=exporters)
        api.upload_folder(repo_id=MODEL_REPO, folder_path=native, path_in_repo=f'comparison/{name}/native_labels',
                          commit_message=f'{name}: {min(start + CHUNK, len(scans))} of {len(scans)} scans')
        memory = psutil.virtual_memory()
        log.info('%s: %d of %d scans predicted and uploaded; memory in use %.0f of %.0f GB (peak of this process %.0f GB)',
                 name, min(start + CHUNK, len(scans)), len(scans), memory.used / 1e9, memory.total / 1e9,
                 resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6)
    lut = np.zeros(256, np.uint8)
    lut[list(mapping)] = list(mapping.values())
    for path in sorted(native.glob('*.nii.gz')):
        image = nib.load(path)
        nib.save(nib.Nifti1Image(lut[np.asanyarray(image.dataobj)], image.affine), output / path.name)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('models', nargs='+', choices=list(MODELS))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    log.info('main_then_release is active: the Colab runtime is released on every exit path')
    console = os.environ.get('ADRENAL_COMPARE_LOG')
    if console:  # the log reaches Hugging Face every 10 minutes, so a machine that vanishes still leaves its trace
        def ship_log():
            while True:
                time.sleep(90)
                try:
                    HfApi().upload_file(repo_id=MODEL_REPO, path_or_fileobj=console, path_in_repo='comparison/compare.log',
                                        commit_message='comparison log (running)')
                except Exception as error:
                    log.warning('log upload failed: %r', error)
        threading.Thread(target=ship_log, daemon=True).start()

        def sample_memory():  # machines have vanished without a trace: leave one every 15 s
            while True:
                memory, disk = psutil.virtual_memory(), psutil.disk_usage('/')
                log.info('memory %.1f of %.1f GB used, %.1f GB available; disk %.0f GB free',
                         memory.used / 1e9, memory.total / 1e9, memory.available / 1e9, disk.free / 1e9)
                time.sleep(15)
        threading.Thread(target=sample_memory, daemon=True).start()
    fetch_test_scans()
    checks = [check_batching(name, fetch_model(name), *MODELS[name][3:]) for name in args.models]
    (DATA / 'comparison').mkdir(exist_ok=True)
    (DATA / 'comparison/batching_check.json').write_text(json.dumps(checks, indent=2))
    HfApi().upload_file(repo_id=MODEL_REPO, path_or_fileobj=DATA / 'comparison/batching_check.json',
                        path_in_repo='comparison/batching_check.json', commit_message='batched sliding window check')
    if not all(check['passed'] for check in checks):
        raise SystemExit("The batched sliding window disagrees with nnU-Net's own: nothing was predicted")
    for name in args.models:
        output = DATA / 'comparison' / name
        log.info('=== %s', name)
        predict(name, fetch_model(name), output, *MODELS[name][3:])
        # evaluate.py checks that every scan has a prediction
        subprocess.run([sys.executable, str(ROOT / 'evaluate.py'), str(RAW / 'labelsTs'), str(output)], check=True)
        HfApi().upload_folder(repo_id=MODEL_REPO, folder_path=output, path_in_repo=f'comparison/{name}',
                              commit_message=f'comparison on the held-out test scans: {name}')
        log.info('Uploaded comparison/%s to %s', name, MODEL_REPO)


def release_runtime(reason):
    """On Colab: upload the console log, then give the (paid) runtime back. Anywhere else: nothing to do."""
    log.info('Exit path: %s', reason)
    try:
        from google.colab import runtime
    except ImportError:
        return log.info('Not on Colab: no runtime to release')
    console = os.environ.get('ADRENAL_COMPARE_LOG')
    try:
        if console:
            HfApi().upload_file(repo_id=MODEL_REPO, path_or_fileobj=console, path_in_repo='comparison/compare.log',
                                commit_message=f'comparison log ({reason})')
    finally:
        log.info('Releasing the Colab runtime')
        runtime.unassign()


def main_then_release():
    """Whatever ends main() - the end, sys.exit, Ctrl-C, a kill or a crash - the Colab runtime is released."""
    signal.signal(signal.SIGTERM, lambda *_: sys.exit('terminated (SIGTERM)'))
    reason = 'finished'
    try:
        main()
    except KeyboardInterrupt:
        reason = 'interrupted (SIGINT)'
        raise
    except SystemExit as stop:
        reason = f'sys.exit({stop.code!r})'
        raise
    except BaseException as error:
        reason = f'crashed: {error!r}'
        raise
    finally:
        release_runtime(reason)


def self_check():
    import numpy as np
    lut = np.zeros(256, np.uint8)
    lut[[8, 9, 50]] = [2, 1, 7]
    assert lut[np.array([0, 8, 9, 50, 60])].tolist() == [0, 2, 1, 7, 0]


if __name__ == '__main__':
    self_check()
    main_then_release()
