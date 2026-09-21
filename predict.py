"""Step 5: segment the held-out test scans with the trained model and score them (evaluate.py).

    python predict.py                 # the final checkpoint (epoch 2000)
    python predict.py --checkpoint best

Reads   data/nnUNet_raw/<dataset>/imagesTs and labelsTs (written by build_dataset.py)
Writes  data/predictions/<checkpoint>/             one segmentation per scan, on the scan's original grid
        data/predictions/<checkpoint>/evaluation/  Dice, surface Dice and volume error (see evaluate.py)
        data/predictions/<checkpoint>/predict.json which checkpoint and settings produced them

If the model is not on this disk it is downloaded from Hugging Face. Prediction is compare.py's: nnU-Net's own
predictor, no mirroring, tile step 0.5, SimpleITK resampling as in the model's plans. Finished scans are skipped.
"""
import argparse
import subprocess
import sys

from compare import RAW, predict
from train import DATA, FOLD, FOLD_FOLDER, MODEL_FOLDER, ROOT, fetch_model_from_hugging_face, has_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--checkpoint', choices=['final', 'best'], default='final')
    checkpoint = f"checkpoint_{parser.parse_args().checkpoint}.pth"
    if not has_checkpoint():
        fetch_model_from_hugging_face()
    if not (FOLD_FOLDER / checkpoint).exists():
        raise SystemExit(f'{FOLD_FOLDER / checkpoint} does not exist (has training finished?)')
    output = DATA / 'predictions' / checkpoint[11:-4]
    predict('ours', MODEL_FOLDER, output, FOLD, checkpoint)
    subprocess.run([sys.executable, str(ROOT / 'evaluate.py'), str(RAW / 'labelsTs'), str(output)], check=True)


if __name__ == '__main__':
    main()
