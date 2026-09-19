"""Step 3: preprocess the dataset for AtlasNet with nnU-Net's own preprocessing.

    python preprocess.py --data data

1. nnU-Net's dataset integrity check (every image/label pair: same shape, spacing, orientation; known labels only).
2. Plans: AtlasNet's plans are transferred with nnU-Net's own move_plans_between_datasets (architecture, 1 mm
   spacing and CT normalization stay AtlasNet's), then the three resampling functions are set to SimpleITK.
3. nnU-Net's preprocessor on every training scan, its patch-sampling index, and the copy of the labels it validates
   against. These are the three things nnU-Net's preprocess_dataset() does; they are called one by one because that
   function still imports distutils, which Python 3.12 no longer has.
4. The AtlasNet checkpoint is downloaded and its checksum verified.

An interrupted run starts again from the beginning of step 3 (nnU-Net's preprocessor clears its output folder).
Needs `pip install -e vendor/nnUNet` (or the pixi environment).
"""
import argparse
import json
import logging
import os
import shutil
import zipfile
from pathlib import Path

from cohort_scan import sha256

DATASET = 'Dataset902_AdrenalMultiorgan'
PLANS = 'AtlasNetPlans'
ATLASNET = {'repo': 'AbdomenAtlas/AtlasNet', 'revision': 'fd03b410350096da9bc743cd77f937aaffc11fcf',
            'archive': 'AbdomenAtlasNetOrgans.zip', 'source_dataset': 'Dataset224_AbdomenAtlas1.1',
            'source_plans': 'nnUNetPlannerResEncL_torchres_isotropic',
            'checkpoint': 'nnUNetTrainer__nnUNetPlannerResEncL_torchres_isotropic__3d_fullres/fold_all/checkpoint_final.pth',
            'checkpoint_sha256': '73ff6cb8e09bbe0c7ad5097d10c281ef4a757c7740ef00b15ee683e907b3e37e'}
os.environ.setdefault('SITK_THREADS', '2')  # per worker process; the workers already fill the CPUs
RESAMPLING = {  # SimpleITK for everything; see PLAN.md
    'resampling_fn_data': 'resample_data_or_seg_to_shape_sitk',
    'resampling_fn_data_kwargs': {'is_seg': False, 'order': 3, 'order_z': 0, 'force_separate_z': False},
    'resampling_fn_seg': 'resample_data_or_seg_to_shape_sitk',
    'resampling_fn_seg_kwargs': {'is_seg': True, 'order': 1, 'order_z': 0, 'force_separate_z': False},
    'resampling_fn_probabilities': 'resample_logits_to_shape_sitk',
    'resampling_fn_probabilities_kwargs': {'is_seg': False, 'order': 1, 'order_z': 0, 'force_separate_z': None}}

log = logging.getLogger('preprocess')


def write_plans(root, preprocessed):
    """AtlasNet's plans under our dataset's name, with SimpleITK resampling."""
    from nnunetv2.experiment_planning.plans_for_pretraining.move_plans_between_datasets import move_plans_between_datasets
    source = preprocessed.parent / ATLASNET['source_dataset']  # move_plans reads the source plans from a dataset folder
    source.mkdir(parents=True, exist_ok=True)
    shutil.copy(root / 'atlasnet_plans.json', source / f"{ATLASNET['source_plans']}.json")
    move_plans_between_datasets(ATLASNET['source_dataset'], DATASET, ATLASNET['source_plans'], PLANS)
    shutil.rmtree(source)
    path = preprocessed / f'{PLANS}.json'
    plans = json.loads(path.read_text())
    plans['configurations']['3d_fullres'] |= RESAMPLING
    path.write_text(json.dumps(plans, indent=2))


def fetch_atlasnet_checkpoint(data):
    from huggingface_hub import hf_hub_download
    target = data / 'atlas/checkpoint_final.pth'
    if not target.exists():
        archive = hf_hub_download(ATLASNET['repo'], ATLASNET['archive'], revision=ATLASNET['revision'])
        with zipfile.ZipFile(archive) as z:
            member = next(n for n in z.namelist() if n.endswith(ATLASNET['checkpoint']))
            target.parent.mkdir(parents=True, exist_ok=True)
            with z.open(member) as source, open(target, 'wb') as out:
                shutil.copyfileobj(source, out)
    if sha256(target) != ATLASNET['checkpoint_sha256']:
        raise SystemExit(f'{target} is not the expected AtlasNet checkpoint')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data', required=True, type=Path, help='the folder build_dataset.py wrote into')
    parser.add_argument('--workers', type=int, default=min(32, os.cpu_count()), help='each worker needs about 4 GB of RAM')
    args = parser.parse_args()
    data = args.data.resolve()
    raw, preprocessed = data / 'nnUNet_raw' / DATASET, data / 'nnUNet_preprocessed' / DATASET
    os.environ['nnUNet_raw'] = str(data / 'nnUNet_raw')
    os.environ['nnUNet_preprocessed'] = str(data / 'nnUNet_preprocessed')
    os.environ['nnUNet_results'] = str(data / 'nnUNet_results')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s',
                        handlers=[logging.StreamHandler(), logging.FileHandler(preprocessed / 'preprocess.log', 'a')])

    from nnunetv2.experiment_planning.verify_dataset_integrity import verify_dataset_integrity
    from nnunetv2.preprocessing.preprocessors.default_preprocessor import DefaultPreprocessor
    from nnunetv2.preprocessing.sampling_locations.extract_sampling_locations import extract_sampling_locations_dataset

    log.info('Checking the dataset with nnU-Net')
    verify_dataset_integrity(str(raw), args.workers)
    write_plans(Path(__file__).resolve().parent, preprocessed)
    shutil.copy(raw / 'dataset.json', preprocessed / 'dataset.json')

    log.info('Preprocessing with %d workers', args.workers)
    DefaultPreprocessor(verbose=False).run(DATASET, '3d_fullres', PLANS, num_processes=args.workers)
    extract_sampling_locations_dataset(DATASET, PLANS, configurations=('3d_fullres',))
    shutil.copytree(raw / 'labelsTr', preprocessed / 'gt_segmentations', dirs_exist_ok=True)

    fetch_atlasnet_checkpoint(data)
    log.info('Finished: %s', preprocessed)


if __name__ == '__main__':
    main()
