"""Existing-VM follow-up: wait for inference, score raw TS masks, publish verified evidence."""
import json
from pathlib import Path
import runpy
import subprocess
import sys
import time

ROOT = Path('/opt/atlasnet-totalseg')


def main():
    while not (ROOT / 'complete.json').exists():
        active = subprocess.run(['pgrep', '-f', '^/opt/flare-run/.pixi/envs/default/bin/python -u /tmp/atlasnet-totalseg.py$'],
                                stdout=subprocess.DEVNULL).returncode == 0
        if not active:
            raise SystemExit('Inference exited without a completion manifest; inspect inference logs')
        time.sleep(60)
    subprocess.run([sys.executable, str(ROOT / 'scoring/evaluate_totalseg.py'), '--run', str(ROOT),
                    '--data', '/opt/adrenal-multiorgan/data', '--workers', '8'], check=True)
    driver = runpy.run_path('/tmp/atlasnet-totalseg.py')
    driver['auth']()
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi()
    output = ROOT / 'evaluation'
    prefix = driver['PREFIX'] + '/evaluation'
    repo = driver['REPO']
    commit = api.upload_folder(repo_id=repo, folder_path=output, path_in_repo=prefix,
                               allow_patterns=['*.csv'], commit_message='Raw TotalSegmentator scoring by Dataset902 split')
    evidence = json.loads((output / 'complete.json').read_text())
    for name, expected in evidence['files'].items():
        fetched = hf_hub_download(repo, f'{prefix}/{name}', revision=commit.oid)
        if driver['sha'](fetched) != expected:
            raise ValueError(f'Uploaded evaluation checksum mismatch: {name}')
    api.upload_file(repo_id=repo, path_or_fileobj=output / 'complete.json', path_in_repo=f'{prefix}/complete.json')
    print('COMPLETE: evaluation uploaded and checksums verified', flush=True)


if __name__ == '__main__':
    main()
