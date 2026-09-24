#!/usr/bin/env bash
# Run compare.py on a Colab A100 with the Colab CLI (Linux/macOS/WSL). Needs a Hugging Face token with write access
# in ~/.cache/huggingface/token or $HF_TOKEN. The run is detached: results are uploaded to Hugging Face by compare.py.
#     bash colab_compare.sh                 start
#     bash colab_compare.sh log             show progress
# compare.py releases the GPU itself on every exit path and uploads its log to comparison/compare.log first.
#     colab stop -s adrenal-compare         only needed to stop a run early
set -euo pipefail
cd "$(dirname "$0")"
SESSION=adrenal-compare
if [ "${1:-}" = keepalive ]; then
    exec ~/.local/share/uv/tools/google-colab-cli/bin/python /tmp/adrenal_keepalive.py
fi
if [ "${1:-}" = poll ]; then  # Colab removed runtimes that nobody talked to for ~20 min; a log read every minute is talking
    while colab sessions 2>/dev/null | grep -q Hardware; do
        bash "$0" log > /tmp/adrenal_poll.log 2>&1 || true
        sleep 60
    done
    exit 0
fi
if [ "${1:-}" = log ]; then
    cat > /tmp/adrenal_log.py <<'PY'
import re
lines = open('/content/compare.log', errors='replace').read().replace('\r', '\n').splitlines()
print('\n'.join(l[:300] for l in lines if re.search(r'scans predicted|Batching check|=== |Exit path|Traceback|Error|Killed|died|verified|Uploaded', l) and 'HTTP' not in l)[-5000:])
PY
    exec colab exec -s $SESSION --timeout 60 -f /tmp/adrenal_log.py
fi
TOKEN=${HF_TOKEN:-$(cat ~/.cache/huggingface/token)}
colab new -s $SESSION --gpu A100 --high-mem || echo 'using the session that is already running'
for file in compare.py evaluate.py; do colab upload -s $SESSION $file /content/$file; done  # newer than the clone
cat > /tmp/adrenal_start.py <<'EOF'
import os, subprocess
run = lambda command: subprocess.run(command, shell=True, check=True, cwd='/content')
run('test -d adrenal-multiorgan || git clone -q https://github.com/ahigazy1/adrenal-multiorgan; git -C adrenal-multiorgan pull -q')
run('cp compare.py evaluate.py adrenal-multiorgan/')
run('pip install -q -e adrenal-multiorgan/vendor/nnUNet surface-distance==0.1 polars cupy-cuda12x cucim-cu12')  # Colab's own torch is kept
if os.environ.get('ADRENAL_GPU_RESAMPLE'):  # the GPU resampler must reproduce nnU-Net's scipy one before it is used
    check = """
import numpy as np
from nnunetv2.preprocessing.resampling.default_resampling import resample_data_or_seg_to_shape as cpu
from nnunetv2.preprocessing.resampling.cucim_resampling import resample_data_or_seg_to_shape_cucim as gpu
x = np.random.default_rng(0).normal(0, 1, (1, 60, 90, 80)).astype(np.float32)
for new, cur, spacing in (((80, 120, 100), (1.5, 1.5, 1.5), (1.125, 1.125, 1.2)), ((150, 110, 90), (5, 0.8, 0.8), (2, 0.65, 0.71))):
    a, b = cpu(x, new, cur, spacing, order=3), gpu(x, new, cur, spacing, order=3)
    d = float(np.abs(a - b).max()); print('cuCIM vs scipy, largest difference', d)
    assert d < 1e-3, d
"""
    open('/content/resampler_check.py', 'w').write(check)
    run('cd adrenal-multiorgan && python /content/resampler_check.py')
environment = os.environ | {'HF_HUB_DISABLE_PROGRESS_BARS': '1', 'nnUNet_def_n_proc': str(os.cpu_count()), 'ADRENAL_COMPARE_LOG': '/content/compare.log'}  # 4 torch threads per export worker; train.py turns torch.compile on
subprocess.Popen('nohup python compare.py ' + os.environ['MODELS'] + ' > /content/compare.log 2>&1', shell=True,
                 cwd='/content/adrenal-multiorgan', env=environment, start_new_session=True)
print('started; follow it with: bash colab_compare.sh log')
EOF
colab exec -s $SESSION --timeout 1200 --env HF_TOKEN="$TOKEN" --env MODELS="${MODELS:-ours atlasnet labmate997-reoriented labmate998}" --env ADRENAL_EXPORTERS="${ADRENAL_EXPORTERS:-0}" --env ADRENAL_GPU_RESAMPLE="${ADRENAL_GPU_RESAMPLE:-}" -f /tmp/adrenal_start.py
# Colab reclaims a runtime nobody talks to. Refresh its idle timer every 5 minutes for as long as it exists.
cat > /tmp/adrenal_keepalive.py <<'PY'
import time
from colab_cli.common import state
while True:
    _, assignments = state.sync_sessions()
    if not assignments:
        break
    for a in assignments:
        state.client.keep_alive_assignment(a.endpoint)
    time.sleep(300)
PY
nohup setsid ~/.local/share/uv/tools/google-colab-cli/bin/python /tmp/adrenal_keepalive.py > /tmp/adrenal_keepalive.log 2>&1 &
echo "keep-alive running (pid $!). It dies with WSL: after a reboot run  bash colab_compare.sh keepalive"
echo "now keep it talking from a shell that stays open:  bash colab_compare.sh poll"
