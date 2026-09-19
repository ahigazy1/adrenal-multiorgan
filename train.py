"""Step 4: train, or continue training. The same command every time:

    python train.py

It decides by itself what to do:
  1. a checkpoint is on this disk              -> continue from it
  2. otherwise, a checkpoint is on Hugging Face -> download it and continue
  3. otherwise                                 -> start from the AtlasNet weights

Every 100 epochs the trainer uploads the latest and best checkpoints and the logs to Hugging Face
(see trainers/adrenal_multiorgan.py). When training and the final validation finish, everything is
uploaded once more. Needs a CUDA GPU and a Hugging Face login with write access (`hf auth login`
or the HF_TOKEN environment variable).
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
DATASET = 'Dataset902_AdrenalMultiorgan'  # written by the preprocessing step
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
os.environ.setdefault('HF_XET_HIGH_PERFORMANCE', '1')  # faster Hugging Face transfers

MODEL_FOLDER = DATA / 'nnUNet_results' / DATASET / f'{TRAINER}__{PLANS}__{CONFIGURATION}'
FOLD_FOLDER = MODEL_FOLDER / f'fold_{FOLD}'
log = logging.getLogger('train')


def sha256(path):
    with open(path, 'rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def has_checkpoint():
    return any(FOLD_FOLDER.glob('checkpoint_*.pth'))


def fetch_model_from_hugging_face():
    from huggingface_hub import snapshot_download
    log.info('No checkpoint on this disk; looking on Hugging Face (%s)', os.environ['ADRENAL_HF_REPO'])
    try:
        snapshot_download(os.environ['ADRENAL_HF_REPO'], local_dir=DATA / 'nnUNet_results' / DATASET,
                          allow_patterns=[f'{MODEL_FOLDER.name}/*'])
    except Exception as error:  # a repository that does not exist yet simply means a first start
        log.info('Nothing downloaded: %r', error)


def run_record():
    """Fingerprint of everything that must stay the same for a resumed run to be the same experiment."""
    preprocessed = DATA / 'nnUNet_preprocessed' / DATASET
    return {'trainer': sha256(ROOT / 'trainers/adrenal_multiorgan.py'), 'plans': sha256(preprocessed / f'{PLANS}.json'),
            'split': sha256(preprocessed / 'splits_final.json'), 'atlasnet_weights': ATLASNET_SHA256,
            'dataset': DATASET, 'fold': FOLD, 'seed': SEED}


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s',
                        handlers=[logging.StreamHandler(), logging.FileHandler(DATA / 'train.log')])
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
        if json.loads(record_path.read_text()) != run_record():
            raise SystemExit(f'The code, plans or split changed since this run started; refusing to continue it. See {record_path}')
        log.info('Continuing from the checkpoint in %s', FOLD_FOLDER)
        weights = None
    else:
        if sha256(ATLASNET_WEIGHTS) != ATLASNET_SHA256:
            raise SystemExit(f'{ATLASNET_WEIGHTS} is not the expected AtlasNet checkpoint')
        log.info('Starting a new run from the AtlasNet weights')
        FOLD_FOLDER.mkdir(parents=True, exist_ok=True)
        record_path.write_text(json.dumps(run_record(), indent=2))
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
