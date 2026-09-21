"""The runtime is released on every exit path of compare.py: python tests/test_compare_exit.py"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PATHS = {'finished': 'pass',
         "sys.exit('check failed')": "raise SystemExit('check failed')",
         "crashed: ValueError('boom')": "raise ValueError('boom')",
         'interrupted (SIGINT)': 'import signal, time; signal.raise_signal(signal.SIGINT); time.sleep(5)'}
for expected, body in PATHS.items():
    code = (f"import compare\ndef main():\n    {body}\ncompare.main = main\n"
            "compare.release_runtime = lambda reason: print('RELEASED:', reason)\ncompare.main_then_release()")
    out = subprocess.run([sys.executable, '-c', code], cwd=ROOT, capture_output=True, text=True).stdout
    assert f'RELEASED: {expected}' in out, (expected, out)
    print('ok:', expected)
