"""Run compare.py on Modal: one GPU container per model, all in parallel, nothing depends on this computer staying on.

    modal run --detach modal_compare.py --dataset raos --models ours-epoch1957,labmate998,labmate997-fixed,atlasnet

Needs the Modal secret 'huggingface-secret' (a Hugging Face token with write access). Results land on Hugging Face
exactly as from Colab: compare.py uploads every 20 scans and resumes from there if a container is restarted.
Modal bills per second: the GPU, plus CPU cores and memory (modal.com/pricing).
"""
import os
import subprocess
import sys

import modal

CPUS, MEMORY_GIB = 32, 128  # exporting (resampling every class of the network output) is CPU work
image = (modal.Image.debian_slim(python_version='3.12')
         .pip_install('torch==2.10.0', index_url='https://download.pytorch.org/whl/cu128')
         .pip_install('huggingface_hub>=1,<2', 'polars', 'surface-distance==0.1', 'psutil', 'scikit-learn', 'openpyxl')
         .add_local_dir('.', '/app', copy=True, ignore=['data', '.git', '**/__pycache__', 'test-results'])
         .run_commands('pip install -e /app/vendor/nnUNet'))
app = modal.App('adrenal-compare', image=image, secrets=[modal.Secret.from_name('huggingface-secret')])
settings = dict(cpu=CPUS, memory=MEMORY_GIB * 1024, timeout=24 * 3600)


def token():
    names = [k for k in os.environ if 'HF' in k or 'HUGGING' in k]
    os.environ.setdefault('HF_TOKEN', next((os.environ[k] for k in names if 'TOKEN' in k), ''))
    assert os.environ['HF_TOKEN'], f'no token in huggingface-secret (keys: {names})'
    os.environ.update(HF_HUB_DISABLE_PROGRESS_BARS='1', nnUNet_def_n_proc=str(CPUS), PYTHONPATH='/app', ADRENAL_CPUS=str(CPUS),
                      ADRENAL_MEMORY_GB=str(int(MEMORY_GIB * 1.07)), ADRENAL_EXPORT_GB='8')


@app.function(gpu='RTX-PRO-6000', **settings)
def compare(model: str, dataset: str):
    token()
    subprocess.run([sys.executable, 'compare.py', model, '--dataset', dataset], cwd='/app', check=True)


@app.local_entrypoint()
def main(models: str = '', dataset: str = 'raos'):
    list(compare.starmap([(m, dataset) for m in models.split(',')]))
