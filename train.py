"""Step 4: train, or continue training. The same command every time:

    python train.py

A local checkpoint is preferred. On a fresh disk, a complete, checksum-verified
Hugging Face snapshot is restored before continuing with nnU-Net's checkpoint
loader (network, optimizer, scaler and epoch), not its pretrained-weights loader.
Without a saved run, training starts from the pinned AtlasNet weights.

Every 100 epochs the trainer uploads an atomic snapshot. A resume can recover
only the last successfully uploaded checkpoint, not later work on a deleted disk.
"""
import hashlib
import json
import logging
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = Path(os.environ.get('ADRENAL_DATA', ROOT / 'data'))
DATASET = 'Dataset902_AdrenalMultiorgan'
PLANS = 'AtlasNetPlans'
CONFIGURATION = '3d_fullres'
FOLD = 0
SEED = 42
TRAINER = 'AdrenalMultiorgan'
ATLASNET_WEIGHTS = DATA / 'atlas/checkpoint_final.pth'
ATLASNET_SHA256 = '73ff6cb8e09bbe0c7ad5097d10c281ef4a757c7740ef00b15ee683e907b3e37e'

os.environ.setdefault('ADRENAL_HF_REPO', 'ahigazy1/adrenal-multiorgan-model')
os.environ['nnUNet_raw'] = str(DATA / 'nnUNet_raw')
os.environ['nnUNet_preprocessed'] = str(DATA / 'nnUNet_preprocessed')
os.environ['nnUNet_results'] = str(DATA / 'nnUNet_results')
os.environ['nnUNet_extTrainer'] = str(ROOT / 'trainers')
sys.path.insert(0, str(ROOT / 'trainers'))
os.environ.setdefault('nnUNet_compile', 'true')
os.environ['MPLBACKEND'] = 'Agg'
os.environ.setdefault('HF_XET_HIGH_PERFORMANCE', '1')

MODEL_FOLDER = DATA / 'nnUNet_results' / DATASET / f'{TRAINER}__{PLANS}__{CONFIGURATION}'
FOLD_FOLDER = MODEL_FOLDER / f'fold_{FOLD}'
log = logging.getLogger('train')


def sha256(path):
    with open(path, 'rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def has_checkpoint():
    return any((FOLD_FOLDER / f'checkpoint_{name}.pth').is_file() for name in ('final', 'latest', 'best'))


def fetch_model_from_hugging_face():
    from hf_cache import restore_model
    log.info('No checkpoint on this disk; looking on Hugging Face (%s)', os.environ['ADRENAL_HF_REPO'])
    return restore_model(MODEL_FOLDER, run_record())


def run_record():
    """Fingerprint of everything that must stay fixed for the same experiment."""
    preprocessed = DATA / 'nnUNet_preprocessed' / DATASET
    record = {'trainer': sha256(ROOT / 'trainers/adrenal_multiorgan.py'), 'plans': sha256(preprocessed / f'{PLANS}.json'),
              'split': sha256(preprocessed / 'splits_final.json'), 'atlasnet_weights': ATLASNET_SHA256,
              'dataset': DATASET, 'fold': FOLD, 'seed': SEED}
    marker = DATA / '.prepared'
    if marker.is_file() and marker.stat().st_size:
        from hf_cache import check_manifest
        value = check_manifest(json.loads(marker.read_text()), 'preprocessed')
        record['preprocessed_content_id'] = value['content_id']
    return record


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s',
                        handlers=[logging.StreamHandler(), logging.FileHandler(DATA / 'train.log', 'a')])
    import numpy as np
    import torch
    from adrenal_multiorgan import upload
    from nnunetv2.run.run_training import run_training
    if not torch.cuda.is_available():
        raise SystemExit('No CUDA GPU found. Training needs the GPU machine.')
    log.info('GPU: %s, torch %s', torch.cuda.get_device_name(), torch.__version__)

    if not has_checkpoint():
        fetch_model_from_hugging_face()
    record_path = FOLD_FOLDER / 'run.json'
    if has_checkpoint():
        if not record_path.is_file() or json.loads(record_path.read_text()) != run_record():
            raise SystemExit(f'The code, plans, split or preprocessing changed; refusing to continue. See {record_path}')
        # Check compatibility even when training is already finished.
        if (FOLD_FOLDER / 'checkpoint_final.pth').exists() and (FOLD_FOLDER / 'validation/summary.json').exists():
            log.info('Training and the final validation have already finished; nothing to do.')
            return
        log.info('Continuing from the checkpoint in %s', FOLD_FOLDER)
        weights = None
    else:
        if sha256(ATLASNET_WEIGHTS) != ATLASNET_SHA256:
            raise SystemExit(f'{ATLASNET_WEIGHTS} is not the expected AtlasNet checkpoint')
        log.info('Starting a new run from the AtlasNet weights')
        FOLD_FOLDER.mkdir(parents=True, exist_ok=True)
        from hf_cache import write_json
        write_json(record_path, run_record())
        weights = str(ATLASNET_WEIGHTS)

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    run_training(DATASET, CONFIGURATION, FOLD, TRAINER, PLANS, pretrained_weights=weights,
                 continue_training=weights is None, device=torch.device('cuda'))

    upload(MODEL_FOLDER, 'training and final validation finished')
    log.info('Finished.')


if __name__ == '__main__':
    main()
