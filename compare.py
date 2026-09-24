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
          'ours-epoch1957': (MODEL_REPO, 'model', MODEL_FOLDER.name, 0, 'checkpoint_best.pth'),  # best of the finished run; 'ours' holds epoch 1300
          'labmate997': ('ahigazy1/adrenal-training-models', 'model', 'labmate', 0, 'checkpoint_final.pth'),
          'labmate998': ('ahigazy1/AdrenalSeg-Sources', 'dataset', D998, 0, 'checkpoint_final.pth'),
          'atlasnet': (ATLASNET['repo'], 'model', None, 'all', 'checkpoint_final.pth'),
          'labmate997-reoriented': ('ahigazy1/adrenal-training-models', 'model', 'labmate', 0, 'checkpoint_final.pth'),
          'labmate997-fixed': ('ahigazy1/adrenal-training-models', 'model', 'labmate', 0, 'checkpoint_final.pth')}
# labmate997's plans.json names the reader NibabelIO, which does not reorient: the AMOS and BTCV scans (not stored RAS)
# reach the network in another orientation. The reorienting reader every other model uses fixes that. nnU-Net takes
# the reader from the plans, so it is set there ('labmate997-reoriented' set it in dataset.json, which is ignored).
READER = {'labmate997-fixed': 'NibabelIOWithReorient'}
NAMES = {'adrenal_left': ['adrenal_left', 'adrenal_gland_left'], 'adrenal_right': ['adrenal_right', 'adrenal_gland_right'],
         'inferior_vena_cava': ['inferior_vena_cava', 'postcava']}  # other organs have the same name everywhere
MAX_CHANGED_FRACTION = 1e-4  # accepted: batching changed 1 to 9 voxels per million on the first check; nnU-Net itself repeats exactly
ABORT_CONDITION = f"labels_changed / voxels > {MAX_CHANGED_FRACTION} (batched against nnU-Net's own sliding window)"
CHUNK = 20  # scans predicted between two uploads
# RAOS (Luo et al. 2024): 413 clinical CTs from one hospital that no compared model trained on - cancer patients (Set1),
# after surgery (Set2) and with organs removed (Set3). Its label ids, found from organ position and volume; no aorta/IVC.
RAOS_REPO, RAOS_FOLDER = 'ahigazy1/AdrenalSeg-Sources', 'RAOS-Real'
RAOS_LABELS = {13: 'adrenal_left', 12: 'adrenal_right', 3: 'kidney_left', 4: 'kidney_right', 1: 'liver', 2: 'spleen',
               8: 'pancreas'}
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


def fetch_raos():
    """All RAOS scans as an nnU-Net test folder, labels renumbered to ours; organs RAOS does not label stay 0."""
    import nibabel as nib
    import numpy as np
    raw = DATA / 'raos'
    if (raw / 'done').exists():
        return raw
    source = Path(snapshot_download(RAOS_REPO, repo_type='dataset', allow_patterns=[f'{RAOS_FOLDER}/*/images*/*',
                                                                                   f'{RAOS_FOLDER}/*/labels*/*']))
    (raw / 'imagesTs').mkdir(parents=True, exist_ok=True)
    (raw / 'labelsTs').mkdir(exist_ok=True)
    lut = np.zeros(256, np.uint8)
    for raos, organ in RAOS_LABELS.items():
        lut[raos] = LABELS[organ]
    swapped = 0
    for label in sorted(source.glob(f'{RAOS_FOLDER}/*/labels*/*.nii.gz')):
        group = 'raos' + label.parent.parent.name.split('(Set')[1][0]  # raos1 cancer, raos2 after surgery, raos3 missing organs
        name = f"{group}_{label.name[:-7].replace('.', '-')}"
        folder = label.parent.parent / label.parent.name.replace('labels', 'images')
        image = next(folder / f for f in (f'{label.name[:-7]}_0000.nii.gz', label.name) if (folder / f).exists())  # Ts vs Tr naming
        (raw / 'imagesTs' / f'{name}_0000.nii.gz').write_bytes(image.read_bytes())
        reference = nib.load(label)
        values = lut[np.asanyarray(reference.dataobj)]
        nib.save(nib.Nifti1Image(values, reference.affine), raw / 'labelsTs' / f'{name}.nii.gz')
        # guard on the label map: the left adrenal must lie on the patient's left of the right one (RAS: smaller x)
        canonical = nib.as_closest_canonical(nib.Nifti1Image(values, reference.affine))
        x = {side: np.argwhere(np.asanyarray(canonical.dataobj) == LABELS[side])[:, 0]
             for side in ('adrenal_left', 'adrenal_right')}
        if len(x['adrenal_left']) and len(x['adrenal_right']) and x['adrenal_left'].mean() > x['adrenal_right'].mean():
            swapped += 1
    scans = len(list((raw / 'labelsTs').glob('*.nii.gz')))
    log.info('RAOS: %d scans; left/right adrenal check failed on %d', scans, swapped)
    if swapped > scans // 20:
        raise SystemExit('The RAOS adrenal label ids look swapped: check RAOS_LABELS')
    (raw / 'done').write_text(str(scans))
    return raw


def fetch_external():
    """Every labelled AMOS22 CT (300) and BTCV (30) scan from the pinned sources prepare.sh uses, labels renumbered to
    ours, named as build_dataset.py names them. No quality exclusion: the 8 scans it excluded are flagged when scoring."""
    import nibabel as nib
    import numpy as np
    from cohort_scan import LABEL_IDS
    raw = DATA / 'external'
    if (raw / 'done').exists():
        return raw
    sources = {'amos': ('MedOtter/amos22-ct-dataset', 'c67f7c01e66277038d87975b03b73ece489a3035',
                        [('train/labelsTr', 'train/imagesTr'), ('valid/labelsVa', 'valid/imagesVa')]),
               'btcv': ('lingheng123/btcv', 'c1728b451a00c054875a0a97d7658d1eaf8362b5',
                        [('RawData/Training/label', 'RawData/Training/img')])}
    (raw / 'imagesTs').mkdir(parents=True, exist_ok=True)
    (raw / 'labelsTs').mkdir(exist_ok=True)
    for source, (repo, revision, folders) in sources.items():
        root = Path(snapshot_download(repo, repo_type='dataset', revision=revision,
                                      allow_patterns=[f'{folder}/*' for pair in folders for folder in pair]))
        lut = np.zeros(256, np.uint8)
        for organ, value in LABEL_IDS[source].items():
            lut[value] = LABELS[organ]
        for labels, images in folders:
            for label in sorted((root / labels).glob('*.nii.gz')):
                case = label.name[:-7]
                name = case if case.lower().startswith(source) else f'{source}_{case}'  # amos_0001, btcv_label0001
                (raw / 'imagesTs' / f'{name}_0000.nii.gz').write_bytes((root / images / label.name.replace('label', 'img')).read_bytes())
                reference = nib.load(label)
                nib.save(nib.Nifti1Image(lut[np.asanyarray(reference.dataobj)], reference.affine), raw / 'labelsTs' / f'{name}.nii.gz')
    scans = len(list((raw / 'labelsTs').glob('*.nii.gz')))
    log.info('AMOS22 + BTCV: %d labelled scans', scans)
    (raw / 'done').write_text(str(scans))
    return raw


def reuse_test_predictions(name, prefix):
    """Scans of this set already predicted in the test-set comparison are copied on Hugging Face, not predicted again."""
    from huggingface_hub import CommitOperationCopy
    api = HfApi()
    have = {f.split('/')[-1] for f in api.list_repo_files(MODEL_REPO) if f.startswith(f'{prefix}/{name}/native_labels/')}
    done = [f.path for f in api.list_repo_tree(MODEL_REPO, f'comparison/{name}/native_labels')
            if f.path.split('/')[-1].startswith(('amos_', 'btcv_')) and f.path.split('/')[-1] not in have]
    if done:
        api.create_commit(MODEL_REPO, [CommitOperationCopy(p, p.replace('comparison/', f'{prefix}/', 1)) for p in done],
                          commit_message=f'{prefix}/{name}: reuse {len(done)} test-set predictions')
    log.info('%s: %d scans reused from the test-set comparison', name, len(done))


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
        predictor.plans_manager.plans['image_reader_writer'] = READER[name]
    configuration = predictor.configuration_manager.configuration
    if configuration['resampling_fn_probabilities'] == 'resample_data_or_seg_to_shape':
        configuration['resampling_fn_probabilities'] = RESAMPLING['resampling_fn_probabilities']
    if configuration['resampling_fn_data'] == 'resample_data_or_seg_to_shape' and os.environ.get('ADRENAL_GPU_RESAMPLE'):
        # the CT on the GPU: a cuCIM port of nnU-Net's own scipy resampler (same interpolation and edge mode, float32),
        # vendored from adrenalSegmentator; the same kwargs as the plans
        configuration['resampling_fn_data'] = 'resample_data_or_seg_to_shape_cucim'
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


def machine():
    """Cores and bytes of memory this process may use: in a container that is its limit, not the host's."""
    cpus = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else os.cpu_count()
    limit = Path('/sys/fs/cgroup/memory.max')
    memory = psutil.virtual_memory().total
    if limit.exists() and limit.read_text().strip().isdigit():
        memory = min(memory, int(limit.read_text()))
    if os.environ.get('ADRENAL_CPUS'):  # a container that shows the host's cores and memory, not its share (Modal)
        cpus, memory = int(os.environ['ADRENAL_CPUS']), int(os.environ['ADRENAL_MEMORY_GB']) * 1e9
    return cpus, memory


def predict(name, model, output, fold, checkpoint, prefix='comparison'):
    import nibabel as nib
    import numpy as np
    predictor = load_predictor(name, model, fold, checkpoint)
    configuration = predictor.configuration_manager.configuration
    mapping = label_mapping(model)
    record = {'model': name, 'checkpoint_sha256': sha256(model / f'fold_{fold}' / checkpoint), 'mirroring': False,
              'tile_step_size': 0.5, 'predictor': 'batched_predictor.py',
              'image_reader': predictor.plans_manager.plans['image_reader_writer'], 'resampling_fn_data': configuration['resampling_fn_data'],
              'resampling_fn_probabilities': configuration['resampling_fn_probabilities'],
              'label_mapping': {str(k): v for k, v in mapping.items()}}
    record_path = output / 'predict.json'
    if record_path.exists() and json.loads(record_path.read_text()) != record:
        raise SystemExit(f'{output} holds predictions of a different model; move it away first')
    native = output / 'native_labels'
    native.mkdir(parents=True, exist_ok=True)
    record_path.write_text(json.dumps(record, indent=2))
    # scans predicted by an earlier, lost machine: fetch them so they are not predicted again
    snapshot_download(MODEL_REPO, allow_patterns=[f'{prefix}/{name}/native_labels/*'], local_dir=DATA)
    api = HfApi()
    # Export workers hold the full-size network output of a scan (classes x voxels, float32, twice), so models with many
    # classes get fewer of them. The workers are new processes and read SITK_THREADS when they start.
    # Exporting (resampling every class of the network output to the scan's grid) is the slow step, so it gets parallel
    # workers: as many as memory allows, two cores each at least. A 66-class export of a whole-body TotalSegmentator scan
    # peaked near 20 GB; RAOS scans are abdomen-only, and 16 workers of the 66-class model fitted in 128 GiB.
    cpus, memory = machine()
    exporters = max(1, min(cpus // 2, int(memory / float(os.environ.get('ADRENAL_EXPORT_GB', 25)) / 1e9)))
    exporters = int(os.environ.get('ADRENAL_EXPORTERS') or 0) or exporters  # a fixed number if one is given
    os.environ['SITK_THREADS'] = str(max(1, cpus // exporters))
    log.info('%s: %d export workers x %s SimpleITK threads on %d cores', name, exporters, os.environ['SITK_THREADS'], cpus)
    scans = sorted((RAW / 'imagesTs').glob('*_0000.nii.gz'))
    for start in range(0, len(scans), CHUNK):  # after every chunk the predictions are safe on Hugging Face
        chunk = [s for s in scans[start:start + CHUNK] if not (native / s.name.replace('_0000', '')).exists()]
        if not chunk:
            continue
        predictor.predict_from_files([[str(s)] for s in chunk], [str(native / s.name[:-12]) for s in chunk],
                                     save_probabilities=False, overwrite=False,
                                     num_processes_preprocessing=max(2, cpus // 4), num_processes_segmentation_export=exporters)
        api.upload_folder(repo_id=MODEL_REPO, folder_path=native, path_in_repo=f'{prefix}/{name}/native_labels',
                          commit_message=f'{name}: {min(start + CHUNK, len(scans))} of {len(scans)} scans')
        memory = psutil.virtual_memory()
        log.info('%s: %d of %d scans predicted and uploaded; memory in use %.0f of %.0f GB',
                 name, min(start + CHUNK, len(scans)), len(scans), memory.used / 1e9, memory.total / 1e9)
    lut = np.zeros(256, np.uint8)
    lut[list(mapping)] = list(mapping.values())
    for path in sorted(native.glob('*.nii.gz')):
        image = nib.load(path)
        nib.save(nib.Nifti1Image(lut[np.asanyarray(image.dataobj)], image.affine), output / path.name)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('models', nargs='+', choices=list(MODELS))
    parser.add_argument('--dataset', choices=['test', 'raos', 'external'], default='test')
    parser.add_argument('--skip-check', action='store_true', help='models that already passed the batching check')
    args = parser.parse_args()
    global RAW
    prefix = {'test': 'comparison', 'raos': 'comparison-raos', 'external': 'comparison-external'}[args.dataset]
    organs = ['--organs', *RAOS_LABELS.values()] if args.dataset == 'raos' else []
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
    if args.dataset == 'raos':
        RAW = fetch_raos()
    elif args.dataset == 'external':
        RAW = fetch_external()
        for name in args.models:
            reuse_test_predictions(name, prefix)
    else:
        fetch_test_scans()
    checks = [] if args.skip_check else [check_batching(name, fetch_model(name), *MODELS[name][3:]) for name in args.models]
    (DATA / prefix).mkdir(exist_ok=True)
    check_file = DATA / prefix / f"batching_check_{'_'.join(args.models)}.json"
    check_file.write_text(json.dumps(checks, indent=2))
    HfApi().upload_file(repo_id=MODEL_REPO, path_or_fileobj=check_file, path_in_repo=f'{prefix}/{check_file.name}',
                        commit_message='batched sliding window check')
    if not all(check['passed'] for check in checks):
        raise SystemExit("The batched sliding window disagrees with nnU-Net's own: nothing was predicted")
    for name in args.models:
        output = DATA / prefix / name
        log.info('=== %s', name)
        predict(name, fetch_model(name), output, *MODELS[name][3:], prefix=prefix)
        # evaluate.py checks that every scan has a prediction
        subprocess.run([sys.executable, str(ROOT / 'evaluate.py'), str(RAW / 'labelsTs'), str(output), *organs], check=True)
        HfApi().upload_folder(repo_id=MODEL_REPO, folder_path=output, path_in_repo=f'{prefix}/{name}',
                              commit_message=f'{prefix}: {name}')
        log.info('Uploaded %s/%s to %s', prefix, name, MODEL_REPO)


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
