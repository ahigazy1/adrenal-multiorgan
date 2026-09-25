"""Read-only local monitor. Python standard library only; Ctrl+C stops monitoring.

    python pseudolabels/monitor.py --once
    python pseudolabels/monitor.py --interval 60

Uses HF_TOKEN from the environment, otherwise reads HF_TOKEN from GCP Secret
Manager using the signed-in gcloud account. Never saves or prints the token.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hf_cache import digest, verify_remote

EXPECTED = {'amos': 300, 'btcv': 30, 'flare': 50}
REPO = 'ahigazy1/adrenal-pseudolabels'


def gcloud(*args):
    binary = shutil.which('gcloud.cmd') or shutil.which('gcloud')
    if not binary:
        raise RuntimeError('gcloud is not on PATH; install/sign in, or use HF_TOKEN with --no-cloud')
    try:
        result = subprocess.run([binary, *args, '--quiet'], capture_output=True,
                                text=True, encoding='utf-8', errors='replace', timeout=45)
    except subprocess.TimeoutExpired:
        raise RuntimeError('gcloud request timed out') from None
    if result.returncode:
        # Secret-access output is deliberately never included in error messages.
        raise RuntimeError(f'gcloud {args[0]} request failed (exit {result.returncode}); check account/project permissions')
    return result.stdout.strip()


def token_for(project, secret):
    token = os.environ.get('HF_TOKEN', '').strip()
    if token:
        return token, 'HF_TOKEN environment variable'
    token = gcloud('secrets', 'versions', 'access', 'latest', f'--secret={secret}', f'--project={project}')
    if not token:
        raise RuntimeError('HF_TOKEN secret is empty')
    return token, f'Secret Manager ({project}/{secret}); memory only'


def get_json(path, token):
    request = Request('https://huggingface.co/' + path,
                      headers={'Authorization': f'Bearer {token}', 'User-Agent': 'adrenal-pseudo-monitor'})
    try:
        with urlopen(request, timeout=30) as response:
            return json.load(response)
    except HTTPError as error:
        raise RuntimeError(f'Hugging Face HTTP {error.code}; check token access and repository/path') from None
    except (URLError, TimeoutError):
        raise RuntimeError('Hugging Face network request failed or timed out') from None


def local_hashes():
    result = {}
    for key, name in [('code', 'pseudolabel.py'), ('runtime_code', 'runtime.py'),
                      ('requirements', 'requirements.txt')]:
        content = (Path(__file__).parent / name).read_bytes()
        # Git converts Windows CRLF to LF when deploying. Accept either encoding.
        result[key] = {hashlib.sha256(value).hexdigest()
                       for value in (content, content.replace(b'\r\n', b'\n'))}
    return result


def matches_code(recipe, hashes):
    return all(recipe.get(key) in values for key, values in hashes.items())


def load_state(path, repo, project, instance, hashes):
    if not path.exists():
        return None
    try:
        saved = json.loads(path.read_text(encoding='utf-8'))
    except (ValueError, OSError):
        raise RuntimeError('Cannot read monitor state; select a new --state file or restore the bookmark') from None
    if any(saved.get(key) != value for key, value in
           [('repo', repo), ('project', project), ('instance', instance)]):
        raise RuntimeError('Saved monitor state is for a different run context; choose a different --state file')
    if not all(set(saved.get('code_hashes', {}).get(key, [])) & values for key, values in hashes.items()):
        raise RuntimeError('Runner code changed since monitoring began; use --prefix to follow the old run explicitly, or a new --state file')
    prefix = saved.get('prefix', '')
    if not re.fullmatch(r'pseudolabels/totalseg-[^/]+/[0-9a-f]{16}', prefix):
        raise RuntimeError('Invalid saved monitor prefix')
    return prefix


def save_state(path, repo, project, instance, hashes, snapshot):
    if not snapshot['hf'].get('prefix'):
        return
    value = {'repo': repo, 'project': project, 'instance': instance,
             'prefix': snapshot['hf']['prefix'], 'checked_at': snapshot['checked_at'],
             'last_verified': snapshot['hf']['verified'],
             'code_hashes': {key: sorted(values) for key, values in hashes.items()}}
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def progress(manifest, entries, prefix):
    recipe = manifest['recipe']
    sources = {name: value[2] for name, value in recipe['sources'].items()}
    if sources != EXPECTED or manifest.get('scans_total') != sum(EXPECTED.values()):
        raise RuntimeError('Manifest is not the labelled AMOS300/BTCV30/FLARE50 run')
    if len(recipe.get('labels', [])) != 66:
        raise RuntimeError('Manifest is not the 66-foreground-class output')
    expected_prefix = f"pseudolabels/totalseg-{recipe['versions']['TotalSegmentator']}/{digest(recipe)[:16]}"
    if prefix != expected_prefix:
        raise RuntimeError('Manifest recipe does not match its HF prefix')
    files = manifest['files']
    if not all(re.fullmatch(r'(amos|btcv|flare)/[^/]+\.nii\.gz', name) for name in files):
        raise RuntimeError('Manifest contains an unexpected source or output path')
    counts = Counter(name.split('/')[0] for name in files)
    if any(counts[name] > expected for name, expected in EXPECTED.items()):
        raise RuntimeError('Manifest exceeds the expected source inventory')
    verify_remote(entries, prefix, manifest)
    complete = manifest.get('complete') is True
    if complete and any(counts[name] != expected for name, expected in EXPECTED.items()):
        raise RuntimeError('Completion flag has missing cases')
    total = len(files)
    return {'state': 'COMPLETE' if complete else 'IN_PROGRESS', 'verified': total,
            'total': sum(EXPECTED.values()), 'percent': round(100 * total / sum(EXPECTED.values()), 1),
            'sources': {name: {'verified': counts[name], 'total': expected}
                        for name, expected in EXPECTED.items()}, 'prefix': prefix}


def hf_snapshot(token, repo, selected_prefix, hashes):
    info = get_json(f'api/datasets/{quote(repo, safe="/")}?blobs=true', token)
    visibility = 'private' if info.get('private') else 'public'
    revision = info['sha']
    entries = {item['rfilename']: SimpleNamespace(size=item.get('size'),
               blob_id=item.get('blobId'), lfs=item.get('lfs')) for item in info['siblings']}
    candidates = [name[:-len('/dataset.json')] for name in entries
                  if re.fullmatch(r'pseudolabels/totalseg-[^/]+/[0-9a-f]{16}/dataset\.json', name)]
    if selected_prefix:
        candidates = [prefix for prefix in candidates if prefix == selected_prefix]
    matching = []
    for prefix in candidates:
        manifest = get_json(f'datasets/{quote(repo, safe="/")}/resolve/{revision}/{prefix}/dataset.json', token)
        if selected_prefix or matches_code(manifest.get('recipe', {}), hashes):
            result = progress(manifest, entries, prefix)
            result['matches_local_code'] = matches_code(manifest['recipe'], hashes)
            matching.append(result)
    if len(matching) > 1:
        raise RuntimeError('Multiple matching recipes; select one with --prefix: ' + ', '.join(p['prefix'] for p in matching))
    if not matching:
        return {'state': 'WAITING_FOR_MANIFEST', 'verified': 0, 'total': sum(EXPECTED.values()),
                'revision': revision, 'visibility': visibility,
                'note': 'No manifest for this code snapshot yet; inference may not have started or the first upload is pending.'}
    return {**matching[0], 'revision': revision, 'visibility': visibility,
            'url': f'https://huggingface.co/datasets/{repo}/tree/{revision}/{matching[0]["prefix"]}'}


def cloud_snapshot(project, instance, zone):
    instances = json.loads(gcloud('compute', 'instances', 'list', f'--project={project}',
                                  f'--filter=name={instance}', '--format=json'))
    instances = [item for item in instances if item['name'] == instance
                 and (not zone or item['zone'].rsplit('/', 1)[-1] == zone)]
    if not instances:
        return {'status': 'NOT_CREATED', 'instance': instance, 'log': []}
    if len(instances) != 1:
        raise RuntimeError('Multiple VM zones match; provide --zone')
    vm = instances[0]
    zone = vm['zone'].rsplit('/', 1)[-1]
    result = {'status': vm['status'], 'instance': instance, 'zone': zone, 'log': []}
    try:
        if vm['status'] == 'RUNNING':
            raw = gcloud('compute', 'instances', 'get-serial-port-output', instance,
                         f'--project={project}', f'--zone={zone}', '--port=1')
        else:
            query = f'resource.type=gce_instance AND resource.labels.instance_id="{vm["id"]}" AND logName:serial_port_1_output'
            rows = json.loads(gcloud('logging', 'read', query, f'--project={project}',
                                    '--freshness=7d', '--limit=100', '--order=desc', '--format=json'))
            raw = '\n'.join(row.get('textPayload', '') for row in reversed(rows))
        lines = [line for line in raw.splitlines()
                 if re.search(r'===|Uploaded \d|verified on Hugging Face|Traceback|Error|ERROR', line)]
        result['log'] = [re.sub(r'hf_[A-Za-z0-9]+', '[redacted]', line)[-800:] for line in lines[-10:]]
    except RuntimeError as error:
        result['log_error'] = str(error)
    return result


def render(snapshot):
    hf = snapshot['hf']
    print(f"\n{snapshot['checked_at']}  {hf['state']}")
    if 'verified' in hf:
        print(f"Verified uploads: {hf['verified']}/{hf['total']}" +
              (f" ({hf['percent']}%)" if 'percent' in hf else ''))
    if hf.get('visibility'):
        print(f"Output repository: {hf['visibility']}")
    for name, count in hf.get('sources', {}).items():
        print(f"  {name.upper():7s} {count['verified']:3d}/{count['total']}")
    for key in ('note', 'error', 'url'):
        if hf.get(key):
            print(hf[key])
    if hf.get('matches_local_code') is False:
        print('Explicit recipe differs from the current local code.')
    vm = snapshot.get('vm', {})
    if vm:
        print(f"VM: {vm.get('status', 'UNAVAILABLE')} {vm.get('instance', '')} {vm.get('zone', '')}")
        if vm.get('error') or vm.get('log_error'):
            print(vm.get('error') or vm['log_error'])
        for line in vm.get('log', []):
            print('  ' + line)
        if vm.get('status') in ('TERMINATED', 'STOPPED') and hf['state'] != 'COMPLETE':
            print('VM is stopped and verified completion is absent; inspect its final log before resuming.')
    print('Uploads advance after the first case, then batches of 25. Case-done logs are not verified uploads.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--project', default='adrenal-seg')
    parser.add_argument('--instance', default='adrenal-pseudo')
    parser.add_argument('--zone')
    parser.add_argument('--secret', default='HF_TOKEN')
    parser.add_argument('--repo', default=REPO)
    parser.add_argument('--prefix', help='Explicit recipe prefix; otherwise match local runner/runtime/requirements hashes')
    parser.add_argument('--state', type=Path, default=ROOT / '.pseudolabel-monitor.json',
                        help='Local run bookmark, saved atomically without credentials')
    parser.add_argument('--interval', type=int, default=60)
    parser.add_argument('--once', action='store_true')
    parser.add_argument('--json', action='store_true', help='One JSON snapshot per poll')
    parser.add_argument('--no-cloud', action='store_true', help='Skip VM/log queries (Secret Manager authentication still works)')
    args = parser.parse_args()
    if args.interval < 10:
        parser.error('--interval must be at least 10 seconds')
    for value in (args.project, args.instance, args.secret, args.zone or 'unset'):
        if not re.fullmatch(r'[A-Za-z0-9_-]+', value):
            parser.error('Project, instance, secret and zone must be simple identifiers')
    if not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+', args.repo):
        parser.error('--repo must be owner/name')
    if args.prefix and not re.fullmatch(r'pseudolabels/totalseg-[^/]+/[0-9a-f]{16}', args.prefix):
        parser.error('--prefix must be pseudolabels/totalseg-VERSION/RECIPE_HASH')
    token, auth_source = token_for(args.project, args.secret)
    if not args.json:
        print(f'Read-only monitor. Authentication: {auth_source}. Ctrl+C to stop.', flush=True)
    hashes = local_hashes()
    prefix = args.prefix or load_state(args.state, args.repo, args.project, args.instance, hashes)
    while True:
        snapshot = {'checked_at': datetime.now(timezone.utc).isoformat(timespec='seconds')}
        try:
            snapshot['hf'] = hf_snapshot(token, args.repo, prefix, hashes)
            # Stay on the same recipe even if another run appears later.
            prefix = snapshot['hf'].get('prefix', prefix)
            save_state(args.state, args.repo, args.project, args.instance, hashes, snapshot)
        except (RuntimeError, ValueError, KeyError, TypeError) as error:
            snapshot['hf'] = {'state': 'ERROR', 'error': str(error)}
        if not args.no_cloud:
            try:
                snapshot['vm'] = cloud_snapshot(args.project, args.instance, args.zone)
            except (RuntimeError, ValueError, KeyError) as error:
                snapshot['vm'] = {'status': 'UNAVAILABLE', 'error': str(error)}
        if args.json:
            print(json.dumps(snapshot), flush=True)
        else:
            render(snapshot)
        if args.once or snapshot['hf']['state'] == 'COMPLETE':
            return 1 if snapshot['hf']['state'] == 'ERROR' else 0
        time.sleep(args.interval)


if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print('\nMonitor stopped. The cloud job was not changed.')
    except (RuntimeError, OSError) as error:
        print(f'Monitor error: {error}', file=sys.stderr)
        sys.exit(1)
