"""Step 3: preprocess the dataset for AtlasNet and publish the training cache on Hugging Face.

    python preprocess.py --data /content/data

1. Plans: AtlasNet's plans are transferred with nnU-Net's own move_plans_between_datasets (architecture, 1 mm
   spacing and CT normalization stay AtlasNet's), then the three resampling functions are set to SimpleITK.
2. Preprocessing: nnU-Net's own per-case function, 100 scans at a time. Each group is uploaded and then deleted
   from this disk, because the whole cache (about 200 GB) does not fit on a Colab runtime.
3. Also uploaded: the plans, dataset.json, the split, the labels nnU-Net validates against, step 1 and 2's tables,
   the held-out test scans, and the AtlasNet checkpoint, so that the training machine needs this one repository only.
   (nnU-Net's foreground-sampling index needs all scans at once; train.py builds it on the training machine.)

Scans that are already in the repository are skipped, so after an interruption just run it again.
Needs `pip install -e vendor/nnUNet` and a Hugging Face login with write access (`hf auth login`).
"""
import argparse
import hashlib
import json
import logging
import os
import shutil
import zipfile
from multiprocessing import get_context
from pathlib import Path

REPOSITORY = 'ahigazy1/adrenal-multiorgan-cache'
DATASET = 'Dataset902_AdrenalMultiorgan'
PLANS = 'AtlasNetPlans'
GROUP_SIZE = 100
ATLASNET = {'repo': 'AbdomenAtlas/AtlasNet', 'revision': 'fd03b410350096da9bc743cd77f937aaffc11fcf',
            'archive': 'AbdomenAtlasNetOrgans.zip', 'source_dataset': 'Dataset224_AbdomenAtlas1.1',
            'source_plans': 'nnUNetPlannerResEncL_torchres_isotropic',
            'checkpoint': 'nnUNetTrainer__nnUNetPlannerResEncL_torchres_isotropic__3d_fullres/fold_all/checkpoint_final.pth',
            'checkpoint_sha256': '73ff6cb8e09bbe0c7ad5097d10c281ef4a757c7740ef00b15ee683e907b3e37e'}
RESAMPLING = {  # SimpleITK for everything; see PLAN.md
    'resampling_fn_data': 'resample_data_or_seg_to_shape_sitk',
    'resampling_fn_data_kwargs': {'is_seg': False, 'order': 3, 'order_z': 0, 'force_separate_z': False},
    'resampling_fn_seg': 'resample_data_or_seg_to_shape_sitk',
    'resampling_fn_seg_kwargs': {'is_seg': True, 'order': 1, 'order_z': 0, 'force_separate_z': False},
    'resampling_fn_probabilities': 'resample_logits_to_shape_sitk',
    'resampling_fn_probabilities_kwargs': {'is_seg': False, 'order': 1, 'order_z': 0, 'force_separate_z': None}}

log = logging.getLogger('preprocess')


def write_plans(root, preprocessed):
    """AtlasNet's plans under our dataset's name, with SimpleITK resampling. Returns the plans as a dict."""
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
    return plans


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
    with open(target, 'rb') as stream:
        if hashlib.file_digest(stream, 'sha256').hexdigest() != ATLASNET['checkpoint_sha256']:
            raise SystemExit(f'{target} is not the expected AtlasNet checkpoint')
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data', required=True, type=Path, help="the folder build_dataset.py wrote into")
    parser.add_argument('--workers', type=int, default=min(24, os.cpu_count()), help='each worker needs about 4 GB of RAM')
    parser.add_argument('--no-upload', action='store_true', help='keep everything on this disk (for a local test)')
    args = parser.parse_args()
    data = args.data.resolve()
    raw, preprocessed = data / 'nnUNet_raw' / DATASET, data / 'nnUNet_preprocessed' / DATASET
    os.environ['nnUNet_raw'] = str(data / 'nnUNet_raw')
    os.environ['nnUNet_preprocessed'] = str(data / 'nnUNet_preprocessed')
    os.environ['nnUNet_results'] = str(data / 'nnUNet_results')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s',
                        handlers=[logging.StreamHandler(), logging.FileHandler(preprocessed / 'preprocess.log', 'a')])

    from huggingface_hub import HfApi
    from nnunetv2.preprocessing.preprocessors.default_preprocessor import DefaultPreprocessor
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
    from nnunetv2.utilities.utils import get_filenames_of_train_images_and_targets
    api = HfApi()

    def upload(folder, path_in_repo, message, **patterns):
        if not args.no_upload:
            api.upload_folder(repo_id=REPOSITORY, repo_type='dataset', folder_path=folder, path_in_repo=path_in_repo,
                              commit_message=message, **patterns)

    plans = write_plans(Path(__file__).resolve().parent, preprocessed)
    # Two things nnU-Net's own preprocessing command does around the per-case work:
    shutil.copy(raw / 'dataset.json', preprocessed / 'dataset.json')
    shutil.copytree(raw / 'labelsTr', preprocessed / 'gt_segmentations', dirs_exist_ok=True)  # used by the final validation
    dataset_json = json.loads((raw / 'dataset.json').read_text())
    plans_manager = PlansManager(plans)
    configuration = plans_manager.get_configuration('3d_fullres')
    output = preprocessed / configuration.data_identifier
    output.mkdir(exist_ok=True)

    in_repository = set() if args.no_upload else set(api.list_repo_files(REPOSITORY, repo_type='dataset'))
    prefix = f'nnUNet_preprocessed/{DATASET}/{configuration.data_identifier}'
    cases = get_filenames_of_train_images_and_targets(str(raw), dataset_json)
    todo = sorted(c for c in cases if f'{prefix}/{c}.pkl' not in in_repository and not (output / f'{c}.pkl').exists())
    log.info('%d training scans, %d still to preprocess, %d workers', len(cases), len(todo), args.workers)

    preprocessor = DefaultPreprocessor(verbose=False)
    for start in range(0, len(todo), GROUP_SIZE):
        group = todo[start:start + GROUP_SIZE]
        jobs = [(str(output / c), cases[c]['images'], cases[c]['label'], plans_manager, configuration, dataset_json)
                for c in group]
        with get_context('spawn').Pool(args.workers) as pool:  # 'spawn', as nnU-Net does
            pool.starmap(preprocessor.run_case_save, jobs)
        upload(output, prefix, f'Preprocessed scans {start + 1}-{start + len(group)} of {len(todo)}')
        log.info('%d of %d scans done', start + len(group), len(todo))
        if not args.no_upload:
            shutil.rmtree(output)  # uploaded; free the disk for the next group
            output.mkdir()

    if args.no_upload:
        return
    fetch_atlasnet_checkpoint(data)
    upload(data / 'atlas', 'atlas', 'AtlasNet checkpoint used for initialization')
    upload(preprocessed, f'nnUNet_preprocessed/{DATASET}', 'Plans, dataset.json, split, validation labels and log',
           allow_patterns=['*.json', '*.log', 'gt_segmentations/*'])
    upload(raw, f'nnUNet_raw/{DATASET}', 'Held-out test scans and the dataset tables',
           allow_patterns=['imagesTs/*', 'labelsTs/*', '*.json', '*.csv', '*.log'])
    log.info('Finished. Cache: https://huggingface.co/datasets/%s', REPOSITORY)


if __name__ == '__main__':
    main()
