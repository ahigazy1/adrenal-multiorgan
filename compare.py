"""Compare models on the 279 held-out test scans (run on any CUDA machine, e.g. a Colab A100).

    python compare.py ours atlasnet labmate997 labmate998      # or a subset; finished scans are skipped

For every model: download it from Hugging Face, segment the test scans with nnU-Net's own predictor
(no mirroring, tile step 0.5), rename its labels to our nine organs, score with evaluate.py and upload
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
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

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
          'atlasnet': (ATLASNET['repo'], 'model', None, 'all', 'checkpoint_final.pth')}
NAMES = {'adrenal_left': ['adrenal_left', 'adrenal_gland_left'], 'adrenal_right': ['adrenal_right', 'adrenal_gland_right'],
         'inferior_vena_cava': ['inferior_vena_cava', 'postcava']}  # other organs have the same name everywhere
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
        path = hf_hub_download(CACHE_REPO, f'{Path(complete).parent.as_posix()}/{part}', repo_type='dataset')
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


def predict(name, model, output, fold, checkpoint):
    import nibabel as nib
    import numpy as np
    import torch
    import nnunetv2.inference.predict_from_raw_data as nnunet
    from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
    # ponytail: the other groups' trainer classes are not in this repository. Only the network is needed and their
    # trainers build the standard one from plans.json, so an unknown trainer name falls back to nnUNetTrainer.
    find = nnunet.recursive_find_trainer_class_by_name
    nnunet.recursive_find_trainer_class_by_name = lambda trainer: find(trainer) or nnUNetTrainer
    predictor = nnunet.nnUNetPredictor(tile_step_size=0.5, use_mirroring=False, perform_everything_on_device=True,
                                       device=torch.device('cuda'))
    predictor.initialize_from_trained_model_folder(str(model), use_folds=(fold,), checkpoint_name=checkpoint)
    configuration = predictor.configuration_manager.configuration
    if configuration['resampling_fn_probabilities'] == 'resample_data_or_seg_to_shape':
        configuration['resampling_fn_probabilities'] = RESAMPLING['resampling_fn_probabilities']
    mapping = label_mapping(model)
    record = {'model': name, 'checkpoint_sha256': sha256(model / f'fold_{fold}' / checkpoint), 'mirroring': False,
              'tile_step_size': 0.5, 'resampling_fn_data': configuration['resampling_fn_data'],
              'resampling_fn_probabilities': configuration['resampling_fn_probabilities'],
              'label_mapping': {str(k): v for k, v in mapping.items()}}
    record_path = output / 'predict.json'
    if record_path.exists() and json.loads(record_path.read_text()) != record:
        raise SystemExit(f'{output} holds predictions of a different model; move it away first')
    native = output / 'native_labels'
    native.mkdir(parents=True, exist_ok=True)
    record_path.write_text(json.dumps(record, indent=2))
    # Export workers hold the full-size network output of a scan (classes x voxels, float32, twice), so models with many
    # classes get fewer of them. The workers are new processes and read SITK_THREADS when they start.
    cpus = os.cpu_count()
    exporters = max(2, cpus // (4 if len(predictor.dataset_json['labels']) <= 40 else 6))
    os.environ['SITK_THREADS'] = str(cpus // exporters)
    predictor.predict_from_files(str(RAW / 'imagesTs'), str(native), save_probabilities=False, overwrite=False,
                                 num_processes_preprocessing=max(3, cpus // 4), num_processes_segmentation_export=exporters)
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
    fetch_test_scans()
    for name in args.models:
        output = DATA / 'comparison' / name
        log.info('=== %s', name)
        predict(name, fetch_model(name), output, *MODELS[name][3:])
        # evaluate.py checks that every scan has a prediction
        subprocess.run([sys.executable, str(ROOT / 'evaluate.py'), str(RAW / 'labelsTs'), str(output)], check=True)
        HfApi().upload_folder(repo_id=MODEL_REPO, folder_path=output, path_in_repo=f'comparison/{name}',
                              commit_message=f'comparison on the held-out test scans: {name}')
        log.info('Uploaded comparison/%s to %s', name, MODEL_REPO)


def self_check():
    import numpy as np
    lut = np.zeros(256, np.uint8)
    lut[[8, 9, 50]] = [2, 1, 7]
    assert lut[np.array([0, 8, 9, 50, 60])].tolist() == [0, 2, 1, 7, 0]


if __name__ == '__main__':
    self_check()
    main()
