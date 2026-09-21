#!/usr/bin/env bash
# Run compare.py on a Colab A100 with the Colab CLI (Linux/macOS/WSL). Needs a Hugging Face token with write access
# in ~/.cache/huggingface/token or $HF_TOKEN. The run is detached: results are uploaded to Hugging Face by compare.py.
#     bash colab_compare.sh                 start
#     bash colab_compare.sh log             show progress
#     colab stop -s adrenal-compare         release the GPU when it has finished
set -euo pipefail
cd "$(dirname "$0")"
SESSION=adrenal-compare
if [ "${1:-}" = log ]; then
    echo "print(open('/content/compare.log').read()[-6000:])" > /tmp/adrenal_log.py
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
run('pip install -q -e adrenal-multiorgan/vendor/nnUNet surface-distance==0.1 polars')  # Colab's own torch is kept
environment = os.environ | {'HF_HUB_DISABLE_PROGRESS_BARS': '1', 'nnUNet_def_n_proc': '4'}  # 4 torch threads per export worker; train.py turns torch.compile on
subprocess.Popen('nohup python compare.py ours atlasnet labmate997 labmate998 > /content/compare.log 2>&1', shell=True,
                 cwd='/content/adrenal-multiorgan', env=environment, start_new_session=True)
print('started; follow it with: bash colab_compare.sh log')
EOF
colab exec -s $SESSION --timeout 1200 --env HF_TOKEN="$TOKEN" -f /tmp/adrenal_start.py
