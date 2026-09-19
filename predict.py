"""Step 5: segment the held-out test scans with the trained model and score them with nnU-Net's evaluator.

    python predict.py                 # the final checkpoint (epoch 2000)
    python predict.py --checkpoint best

Reads   data/nnUNet_raw/<dataset>/imagesTs and labelsTs (written by build_dataset.py)
Writes  data/predictions/<checkpoint>/  one segmentation per scan, on the scan's original grid
        data/predictions/<checkpoint>/summary.json   Dice and voxel counts per scan and label (nnU-Net's evaluator)
        data/predictions/<checkpoint>/predict.json   which checkpoint and settings produced them

If the model is not on this disk it is downloaded from Hugging Face. Prediction is nnU-Net's own predictor:
no mirroring, tile step 0.5. Resampling follows the model's plans, i.e. SimpleITK for the CT and for the
network output. Already finished scans are skipped, so an interrupted run can simply be started again.
"""
import argparse
import json
import logging

from train import DATA, DATASET, FOLD, FOLD_FOLDER, MODEL_FOLDER, fetch_model_from_hugging_face, has_checkpoint, sha256

log = logging.getLogger('train')  # the same logger train.py's helpers use


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--checkpoint', choices=['final', 'best'], default='final')
    args = parser.parse_args()
    checkpoint = f'checkpoint_{args.checkpoint}.pth'
    raw = DATA / 'nnUNet_raw' / DATASET
    output = DATA / 'predictions' / args.checkpoint
    output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s',
                        handlers=[logging.StreamHandler(), logging.FileHandler(output / 'predict.log')])

    import torch
    from nnunetv2.evaluation.evaluate_predictions import compute_metrics_on_folder2
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    if not torch.cuda.is_available():
        raise SystemExit('No CUDA GPU found. Prediction needs the GPU machine.')
    if not has_checkpoint():
        fetch_model_from_hugging_face()
    if not (FOLD_FOLDER / checkpoint).exists():
        raise SystemExit(f'{FOLD_FOLDER / checkpoint} does not exist (has training finished?)')

    # One model per output folder: refuse to mix predictions from two different checkpoints.
    record = {'checkpoint': checkpoint, 'checkpoint_sha256': sha256(FOLD_FOLDER / checkpoint),
              'plans_sha256': sha256(MODEL_FOLDER / 'plans.json'), 'mirroring': False, 'tile_step_size': 0.5}
    record_path = output / 'predict.json'
    if record_path.exists() and json.loads(record_path.read_text()) != record:
        raise SystemExit(f'{output} holds predictions from a different model; move it away first')
    record_path.write_text(json.dumps(record, indent=2))

    scans = sorted((raw / 'imagesTs').glob('*_0000.nii.gz'))
    log.info('Predicting %d test scans with %s', len(scans), checkpoint)
    predictor = nnUNetPredictor(tile_step_size=0.5, use_mirroring=False, perform_everything_on_device=True,
                                device=torch.device('cuda'))
    predictor.initialize_from_trained_model_folder(str(MODEL_FOLDER), use_folds=(FOLD,), checkpoint_name=checkpoint)
    predictor.predict_from_files(str(raw / 'imagesTs'), str(output), save_probabilities=False, overwrite=False,
                                 num_processes_preprocessing=8, num_processes_segmentation_export=8)

    missing = [s.name for s in scans if not (output / s.name.replace('_0000.nii.gz', '.nii.gz')).exists()]
    if missing:
        raise SystemExit(f'{len(missing)} scans have no prediction, e.g. {missing[:3]}')
    compute_metrics_on_folder2(str(raw / 'labelsTs'), str(output), str(MODEL_FOLDER / 'dataset.json'),
                               str(MODEL_FOLDER / 'plans.json'), str(output / 'summary.json'), num_processes=8)
    means = json.loads((output / 'summary.json').read_text())['mean']
    names = {str(value): name for name, value in json.loads((MODEL_FOLDER / 'dataset.json').read_text())['labels'].items()}
    for label, metrics in means.items():
        log.info('%-20s mean Dice %.4f', names.get(str(label), label), metrics['Dice'])
    log.info('Wrote %s', output)


if __name__ == '__main__':
    main()
