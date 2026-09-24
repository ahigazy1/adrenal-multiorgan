"""Run compare.py on Modal: one GPU container per model, all in parallel, nothing depends on this computer staying on.

    modal run --detach modal_compare.py --dataset raos --models ours-epoch1957,labmate998,labmate997-fixed,atlasnet
    modal run modal_compare.py --benchmark ours-epoch1957,labmate998     # time 6 RAOS scans on each GPU type, no uploads

Needs the Modal secret 'huggingface-secret' (a Hugging Face token with write access). Results land on Hugging Face
exactly as from Colab: compare.py uploads every 20 scans and resumes from there if a container is restarted.
Modal bills per second: the GPU, plus CPU cores and memory (modal.com/pricing).
"""
import os
import subprocess
import sys
import time

import modal

CPUS, MEMORY_GIB = 32, 128  # exporting (resampling every class of the network output) is CPU work
SCANS = 20  # benchmark length: enough that the export workers are busy the whole time
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


def benchmark(model: str) -> dict:
    """Seconds per scan for 6 RAOS scans, end to end (reading, network, export), with compare.py's worker settings."""
    token()
    sys.path.insert(0, '/app')
    os.chdir('/app')
    import compare as c
    import torch
    raw = c.fetch_raos()
    folder = c.fetch_model(model)
    predictor = c.load_predictor(model, folder, *c.MODELS[model][3:])
    scans = sorted((raw / 'imagesTs').glob('raos1_*_0000.nii.gz'))[:SCANS + 1]
    out = c.DATA / 'benchmark'
    cpus, memory = c.machine()
    exporters = max(1, min(cpus // 2, int(memory / 25e9)))
    os.environ['SITK_THREADS'] = str(max(1, cpus // exporters))
    run = lambda group: predictor.predict_from_files([[str(s)] for s in group], [str(out / s.name[:-12]) for s in group],
                                                     save_probabilities=False, overwrite=True,
                                                     num_processes_preprocessing=max(2, cpus // 4),
                                                     num_processes_segmentation_export=exporters)
    run(scans[:1])  # warm-up: torch.compile and cuDNN autotuning
    start = time.time()
    run(scans[1:])
    return {'model': model, 'gpu': torch.cuda.get_device_name(), 'seconds_per_scan': round((time.time() - start) / SCANS, 1),
            'exporters': exporters, 'cpus': cpus, 'memory_gb': round(memory / 1e9)}


@app.function(gpu='RTX-PRO-6000', **settings)
def bench_pro(model: str) -> dict:
    return benchmark(model)


@app.function(gpu='A100-80GB', **settings)
def bench_a100(model: str) -> dict:
    return benchmark(model)


@app.function(gpu='L40S', **settings)
def bench_l40s(model: str) -> dict:
    return benchmark(model)


@app.function(gpu='L4', **settings)
def bench_l4(model: str) -> dict:
    return benchmark(model)


@app.local_entrypoint()
def main(models: str = '', dataset: str = 'raos', benchmark: str = ''):
    if benchmark:
        calls = [f.spawn(m) for m in benchmark.split(',') for f in (bench_pro, bench_a100, bench_l40s, bench_l4)]
        for call in calls:
            print(call.get())
        return
    list(compare.starmap([(m, dataset) for m in models.split(',')]))
