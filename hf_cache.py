"""Verified preprocessing caches and atomic training snapshots on Hugging Face.

Run inside the pinned environment: python hf_cache.py ensure --data data
No upload is considered complete until its manifest is committed. Authentication,
network, integrity and compatibility failures stop the run; they are not cache misses.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import hashlib
import json
import logging
import os
from pathlib import Path, PurePosixPath
import subprocess
import time
import tarfile
from typing import Any

ROOT = Path(__file__).resolve().parent
DATASET = 'Dataset902_AdrenalMultiorgan'
PLANS = 'AtlasNetPlans'
VERSION = 1
SNAPSHOT = '_hf_snapshot.json'
ATLASNET_SHA256 = '73ff6cb8e09bbe0c7ad5097d10c281ef4a757c7740ef00b15ee683e907b3e37e'
CACHE_REPO = 'ahigazy1/adrenal-multiorgan-cache'
CACHE_ATTRIBUTES = ''.join(f'preprocessed/**/*.{extension} filter=lfs diff=lfs merge=lfs -text\n'
                           for extension in ('tar',))
log = logging.getLogger('hf_cache')


def encoded(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(',', ':')).encode()


def digest(value: Any) -> str:
    return hashlib.sha256(encoded(value)).hexdigest()


def file_info(path: Path) -> dict:
    """Hash in bounded memory, including the Git blob hash for non-LFS files."""
    before = path.stat()
    sha = hashlib.sha256()
    git = hashlib.sha1(f'blob {before.st_size}\0'.encode())
    with path.open('rb') as stream:
        while chunk := stream.read(4 * 1024 * 1024):
            sha.update(chunk)
            git.update(chunk)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f'File changed while hashing: {path}')
    return {'size': before.st_size, 'sha256': sha.hexdigest(), 'git_sha1': git.hexdigest()}


def safe_path(root: Path, name: str) -> Path:
    p = PurePosixPath(name)
    if not name or p.is_absolute() or '..' in p.parts or str(p) != name or '\\' in name or ':' in name:
        raise ValueError(f'Unsafe artifact path: {name!r}')
    out = root / name
    if not out.resolve().is_relative_to(root.resolve()) or out.is_symlink():
        raise ValueError(f'Artifact path escapes its destination: {name!r}')
    return out


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('wb') as stream:
        stream.write(encoded(value))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def recipe_id(root: Path = ROOT) -> str:
    """Conservative invalidation for data rules, sources, environment and nnU-Net."""
    names = ['cohort_scan.py', 'build_dataset.py', 'preprocess.py', 'prepare.sh',
             'atlasnet_plans.json', 'pixi.lock']
    vendor = root / 'vendor/nnUNet'
    if not vendor.is_dir():
        raise RuntimeError(f'Missing vendored nnU-Net: {vendor}')
    names += [p.relative_to(root).as_posix() for p in sorted(vendor.rglob('*.py'))]
    return digest({'version': VERSION, 'sources': {n: file_info(root / n)['sha256'] for n in names}})


def manifest(kind: str, recipe: str, files: dict) -> dict:
    result = {'version': VERSION, 'kind': kind, 'recipe': recipe, 'files': files}
    result['content_id'] = digest(result)
    return result


def check_manifest(value: dict, kind: str, recipe: str | None = None) -> dict:
    if not isinstance(value, dict) or value.get('version') != VERSION or value.get('kind') != kind:
        raise RuntimeError('Unsupported artifact manifest')
    if recipe is not None and value.get('recipe') != recipe:
        raise RuntimeError('The preprocessing recipe changed; refusing this cache')
    files = value.get('files')
    if not isinstance(files, dict) or not files:
        raise RuntimeError('Empty artifact manifest')
    if manifest(kind, value.get('recipe'), files)['content_id'] != value.get('content_id'):
        raise RuntimeError('Artifact manifest checksum mismatch')
    for name, info in files.items():
        safe_path(Path('/artifact'), name)
        if kind == 'preprocessed' and not any(name.startswith(prefix) for prefix in (
                f'nnUNet_preprocessed/{DATASET}/', f'nnUNet_raw/{DATASET}/', 'atlas/')):
            raise RuntimeError(f'Unexpected cache path: {name}')
        if (not isinstance(info, dict) or not isinstance(info.get('size'), int) or info['size'] < 0
                or not isinstance(info.get('sha256'), str) or len(info['sha256']) != 64
                or not isinstance(info.get('git_sha1'), str) or len(info['git_sha1']) != 40):
            raise RuntimeError(f'Invalid file metadata: {name}')
    return value


RETRY_SECONDS = 60  # pause before a failed checkpoint upload is tried again (grows with each attempt)
HASH_THREADS = 16  # hashlib releases the GIL, so threads hash in parallel; beyond this the disk is the limit


def hash_files(paths: list[Path]) -> list[dict]:
    with ThreadPoolExecutor(max_workers=HASH_THREADS) as workers:
        return list(workers.map(file_info, paths))


def matches_local(path: Path, info: dict) -> bool:
    return path.is_file() and path.stat().st_size == info['size'] and file_info(path) == info


def verify_local(root: Path, value: dict) -> None:
    paths = [safe_path(root, name) for name in value['files']]
    for path in paths:
        if not path.is_file():
            raise RuntimeError(f'Missing artifact: {path}')
    for path, actual, expected in zip(paths, hash_files(paths), value['files'].values()):
        if actual != expected:
            raise RuntimeError(f'Missing or corrupt artifact: {path}')
    log.info('Verified %d files', len(paths))


def cache_files(data: Path) -> list[Path]:
    """Require every case, every split, the sampling store, GT, test set and weights."""
    pre = data / 'nnUNet_preprocessed' / DATASET
    raw = data / 'nnUNet_raw' / DATASET
    dataset = json.loads((pre / 'dataset.json').read_text())
    plans = json.loads((pre / f'{PLANS}.json').read_text())
    config = safe_path(pre, plans['configurations']['3d_fullres']['data_identifier'])
    if config.parent != pre:
        raise RuntimeError('Unexpected configuration directory')
    with (raw / 'build_dataset.csv').open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    if not rows or any(r['role'] not in ('train', 'test') for r in rows):
        raise RuntimeError('Dataset build has failed or unrecognized cases')
    train = {r['name'] for r in rows if r['role'] == 'train'}
    test = {r['name'] for r in rows if r['role'] == 'test'}
    counts = json.loads((raw / 'build_dataset.json').read_text())['counts']
    if (len(train) + len(test) != len(rows) or train & test or not test or not train
            or len(train) != dataset['numTraining'] or counts.get('failed', 0)
            or counts['train'] != len(train) or counts['test'] != len(test)):
        raise RuntimeError('Dataset counts or case identities do not agree')
    splits = json.loads((pre / 'splits_final.json').read_text())
    if len(splits) != 5:
        raise RuntimeError('Expected five development folds')
    for fold in splits:
        a, b = set(fold['train']), set(fold['val'])
        if not a or not b or a & b or a | b != train or len(a) != len(fold['train']) or len(b) != len(fold['val']):
            raise RuntimeError('Incomplete, overlapping or incompatible training/validation split')
    if {p.stem for p in config.glob('*.pkl')} != train:
        raise RuntimeError('Preprocessed case list is incomplete or has unexpected cases')
    fg = config / 'fg_sampling'
    identifiers = json.loads((fg / 'meta.json').read_text())['identifiers']
    if set(identifiers) != train or len(identifiers) != len(train):
        raise RuntimeError('Foreground sampling index is incomplete')
    required = [pre / f'{PLANS}.json', pre / 'dataset.json', pre / 'dataset_fingerprint.json', pre / 'splits_final.json',
                raw / 'dataset.json', raw / 'build_dataset.csv', raw / 'build_dataset.json',
                data / 'atlas/checkpoint_final.pth']
    required += [fg / n for n in ('locations.b2nd', 'indptr.npy', 'class_id.npy', 'count.npy',
                                  'base.npy', 'shape.npy', 'meta.json')]
    for case in train:
        required += [safe_path(config, case + suffix) for suffix in ('.b2nd', '_seg.b2nd', '.pkl')]
        required.append(safe_path(pre / 'gt_segmentations', case + '.nii.gz'))
    for case in test:
        required += [safe_path(raw / 'imagesTs', case + '_0000.nii.gz'),
                     safe_path(raw / 'labelsTs', case + '.nii.gz')]
    for path in required:
        if path.is_symlink() or not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f'Incomplete preprocessing bundle: {path}')
    # Include any additional nnU-Net metadata, but never logs, temporary files or raw training CTs.
    required += [p for p in pre.rglob('*') if p.is_file() and not p.name.startswith('.')
                 and p.suffix not in ('.log', '.tmp', '.lock') and not any(x.startswith('.') for x in p.relative_to(pre).parts)]
    for path in required:
        safe_path(data, path.relative_to(data).as_posix())
    return sorted(set(required))


def build_cache_manifest(data: Path, recipe: str) -> dict:
    paths = cache_files(data)
    files = {p.relative_to(data).as_posix(): info for p, info in zip(paths, hash_files(paths))}
    if files['atlas/checkpoint_final.pth']['sha256'] != ATLASNET_SHA256:
        raise RuntimeError('AtlasNet weights failed their pinned checksum')
    return manifest('preprocessed', recipe, files)


def cache_prefix(value: dict) -> str:
    return f"preprocessed/v{VERSION}/{value['recipe']}/{value['content_id']}"


def complete_path(recipe: str) -> str:
    return f'preprocessed/v{VERSION}/{recipe}/complete.json'


class Hub:
    """Small transport boundary so failure/recovery behavior can be tested offline."""
    def __init__(self, repo: str, repo_type: str):
        from huggingface_hub import HfApi
        self.api = HfApi()
        self.repo, self.repo_type = repo, repo_type
        # Also fail early on a bad token. Never translate 401/403/404 into a cache miss.
        self.api.create_repo(repo, repo_type=repo_type, private=True, exist_ok=True)

    def info(self):
        result = self.api.repo_info(self.repo, repo_type=self.repo_type, files_metadata=True)
        if not result.private:
            raise RuntimeError(f'Refusing to use a public artifact repository: {self.repo}')
        return result.sha, {p.rfilename: p for p in result.siblings}

    def get(self, name: str, revision: str, directory: Path, force: bool = False) -> Path:
        from huggingface_hub import hf_hub_download
        return Path(hf_hub_download(self.repo, name, repo_type=self.repo_type, revision=revision,
                                   local_dir=directory, force_download=force))

    def commit(self, additions: dict, message: str, parent: str) -> str:
        from huggingface_hub import CommitOperationAdd
        operations = [CommitOperationAdd(path_in_repo=name, path_or_fileobj=str(value) if isinstance(value, Path) else value)
                      for name, value in additions.items()]
        result = self.api.create_commit(self.repo, repo_type=self.repo_type, operations=operations,
                                       commit_message=message, parent_commit=parent)
        return result.oid


def matches_remote(entry, expected: dict) -> bool:
    if entry is None or entry.size != expected['size']:
        return False
    lfs = getattr(entry, 'lfs', None)
    if lfs is not None:
        checksum = lfs.get('sha256') if isinstance(lfs, dict) else lfs.sha256
        return checksum == expected['sha256']
    return entry.blob_id == expected['git_sha1']


def verify_remote(entries: dict, prefix: str, value: dict) -> None:
    for name, info in value['files'].items():
        if not matches_remote(entries.get(f'{prefix}/{name}'), info):
            raise RuntimeError(f'Incomplete or corrupt Hugging Face payload: {prefix}/{name}')


def restore_files(hub: Hub, revision: str, prefix: str, value: dict, target: Path, staging: Path) -> None:
    """Move each verified download into place; interrupted transfers reuse completed files."""
    def one(name: str, info: dict):
        destination = safe_path(target, name)
        if matches_local(destination, info):
            return
        downloaded = hub.get(f'{prefix}/{name}', revision, staging)
        if not matches_local(downloaded, info):
            downloaded = hub.get(f'{prefix}/{name}', revision, staging, force=True)
            if not matches_local(downloaded, info):
                raise RuntimeError(f'Download checksum failed: {name}')
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.replace(downloaded, destination)
    with ThreadPoolExecutor(max_workers=8) as workers:
        jobs = [workers.submit(one, name, info) for name, info in value['files'].items()]
        for index, job in enumerate(as_completed(jobs), 1):
            job.result()
            if index % 50 == 0 or index == len(jobs):
                log.info('Restored %d/%d files', index, len(jobs))


def archive_groups(files: dict, limit: int = 10 * 2**30):
    """Bound extra disk usage to one roughly 10 GiB uncompressed tar at a time."""
    group, size = [], 0
    for name, info in sorted(files.items()):
        if group and size + info['size'] > limit:
            yield group
            group, size = [], 0
        group.append(name)
        size += info['size']
    if group:
        yield group


def cache_archives(value: dict) -> dict:
    archives = value.get('archives')
    if not isinstance(archives, dict) or not archives:
        raise RuntimeError('Cache has no tar archive manifest')
    for name in archives:
        if PurePosixPath(name).name != name or not name.endswith('.tar'):
            raise RuntimeError(f'Unexpected archive name: {name}')
        safe_path(Path('/artifact'), name)
    return archives


def restore_cache(hub: Hub, revision: str, value: dict, data: Path) -> None:
    """Download and verify each tar before extracting; verify all files before training."""
    for name, info in cache_archives(value).items():
        archive = hub.get(f'{cache_prefix(value)}/{name}', revision, data / '.hf-downloads')
        if not matches_local(archive, info):
            raise RuntimeError(f'Download checksum failed: {name}')
        with tarfile.open(archive, 'r') as tar:
            for member in tar:
                safe_path(data, member.name)
                if not member.isfile() or member.name not in value['files']:
                    raise RuntimeError(f'Unexpected tar member: {member.name}')
                tar.extract(member, data, filter='data')
        archive.unlink()
    verify_local(data, value)


def publish_cache(hub: Hub, data: Path, value: dict) -> None:
    """Upload bounded tar archives; publish complete.json after remote verification."""
    revision, entries = hub.info()
    pointer = complete_path(value['recipe'])
    if pointer in entries:
        existing = json.loads(hub.get(pointer, revision, data / '.hf-metadata').read_text())
        check_manifest(existing, 'preprocessed', value['recipe'])
        if existing['content_id'] != value['content_id']:
            raise RuntimeError('A different complete cache exists for this recipe; refusing to overwrite it')
        verify_remote(entries, cache_prefix(existing), {'files': cache_archives(existing)})
        return
    # Explicitly route custom .b2nd arrays and other large binaries through LFS/Xet.
    # Preserve existing rules, including those for the source archive in this repository.
    attributes = (hub.get('.gitattributes', revision, data / '.hf-metadata').read_text()
                  if '.gitattributes' in entries else '')
    missing_rules = [line for line in CACHE_ATTRIBUTES.splitlines() if line not in attributes.splitlines()]
    if missing_rules:
        attributes = attributes.rstrip() + '\n' + '\n'.join(missing_rules) + '\n'
        revision = hub.commit({'.gitattributes': attributes.encode()}, 'Track preprocessing binaries with LFS', revision)
    prefix = cache_prefix(value)
    archives = {}
    archive = data / '.preprocessed-upload.tar'
    for index, batch in enumerate(archive_groups(value['files'])):
        name = f'part-{index:05d}.tar'
        with tarfile.open(archive, 'w') as tar:
            for member in batch:
                tar.add(safe_path(data, member), arcname=member, recursive=False)
        archives[name] = file_info(archive)
        if not matches_remote(entries.get(f'{prefix}/{name}'), archives[name]):
            log.info('Uploading %s (%.1f GiB)', name, archives[name]['size'] / 2**30)
            revision = hub.commit({f'{prefix}/{name}': archive}, f'Preprocessing cache: {name}', revision)
        archive.unlink()
    latest, entries = hub.info()
    if latest != revision:
        raise RuntimeError('Cache repository changed concurrently; rerun to reconcile safely')
    verify_remote(entries, prefix, {'files': archives})
    hub.commit({pointer: encoded(value | {'archives': archives})},
               'Preprocessing tar cache complete and verified', revision)
    log.info('Published complete preprocessing cache: %s', value['content_id'])


def ensure_prepared(data: Path, root: Path = ROOT, hub=None, publish: bool = True) -> None:
    """Make the prepared data available locally. With publish=False nothing is uploaded here: training does not
    need the upload, so startup.sh runs `publish` in the background while the GPU trains."""
    data.mkdir(parents=True, exist_ok=True)
    recipe = recipe_id(root)
    hub = hub or Hub(os.environ.get('ADRENAL_CACHE_REPO', CACHE_REPO), 'dataset')
    for marker in (data / '.prepared', data / '.preprocessed-local.json'):
        if marker.is_file() and marker.stat().st_size:
            value = check_manifest(json.loads(marker.read_text()), 'preprocessed', recipe)
            if publish:
                log.info('Checking locally completed preprocessing')
                verify_local(data, value)
            cache_files(data)  # every expected file is present; same disk, so no full re-hash on each start
            if publish:
                publish_cache(hub, data, value)
            write_json(data / '.prepared', value)
            return
    revision, entries = hub.info()
    pointer = complete_path(recipe)
    if pointer in entries:
        value = check_manifest(json.loads(hub.get(pointer, revision, data / '.hf-metadata').read_text()),
                               'preprocessed', recipe)
        verify_remote(entries, cache_prefix(value), {'files': cache_archives(value)})
        log.info('Complete Hugging Face cache found; skipping downloads of source datasets and preprocessing')
        restore_cache(hub, revision, value, data)
        cache_files(data)
    else:
        log.info('No complete compatible cache; preparing locally with prepare.sh')
        subprocess.run(['bash', str(root / 'prepare.sh'), str(data)], cwd=root, check=True)
        value = build_cache_manifest(data, recipe)
        write_json(data / '.preprocessed-local.json', value)
        if publish:
            publish_cache(hub, data, value)
    write_json(data / '.prepared', value)
    log.info('Verified preprocessing is ready for training')


def publish_prepared(data: Path, hub=None) -> None:
    """Upload the prepared data if Hugging Face does not have it yet. Safe to run beside training and to repeat."""
    value = check_manifest(json.loads((data / '.prepared').read_text()), 'preprocessed', recipe_id())
    publish_cache(hub or Hub(os.environ.get('ADRENAL_CACHE_REPO', CACHE_REPO), 'dataset'), data, value)


def publish_model(model_folder: Path, message: str, hub=None) -> str:
    """Checkpoints, settings and a manifest become visible in one atomic Hub commit."""
    files = {p.relative_to(model_folder).as_posix(): file_info(p) for p in sorted(model_folder.rglob('*'))
             if p.is_file() and p.name != SNAPSHOT and p.suffix not in ('.tmp', '.lock')
             and not any(x.startswith('.') for x in p.relative_to(model_folder).parts)}
    if 'fold_0/run.json' not in files or not any(f'fold_0/checkpoint_{n}.pth' in files for n in ('final', 'latest', 'best')):
        raise RuntimeError('Cannot publish a model without its checkpoint and run record')
    value = manifest('model', files['fold_0/run.json']['sha256'], files)
    hub = hub or Hub(os.environ['ADRENAL_HF_REPO'], 'model')
    revision, _ = hub.info()
    additions = {f'{model_folder.name}/{name}': safe_path(model_folder, name) for name in files}
    additions[f'{model_folder.name}/{SNAPSHOT}'] = encoded(value)
    # One failed attempt must not mean "no backup for the next 100 epochs": try again with growing pauses.
    for attempt in range(1, 5):
        try:
            return hub.commit(additions, f'{model_folder.name}: {message}', revision)
        except Exception as error:
            if attempt == 4:
                raise
            log.warning('Checkpoint upload failed (attempt %d of 4), retrying: %r', attempt, error)
            time.sleep(RETRY_SECONDS * attempt)
            revision, _ = hub.info()


def restore_model(model_folder: Path, expected_record: dict, hub=None) -> bool:
    """Do not expose any downloaded checkpoint until the entire snapshot is valid."""
    hub = hub or Hub(os.environ['ADRENAL_HF_REPO'], 'model')
    revision, entries = hub.info()
    prefix = model_folder.name
    pointer = f'{prefix}/{SNAPSHOT}'
    staging = model_folder.parent / '.hf-model-restore'
    if pointer not in entries:
        if any(n.startswith(prefix + '/') for n in entries):
            raise RuntimeError('Model files exist on Hugging Face without a complete snapshot manifest; refusing to start over')
        return False
    value = check_manifest(json.loads(hub.get(pointer, revision, staging).read_text()), 'model')
    verify_remote(entries, prefix, value)
    record_name = 'fold_0/run.json'
    if record_name not in value['files'] or not any(f'fold_0/checkpoint_{n}.pth' in value['files'] for n in ('final', 'latest', 'best')):
        raise RuntimeError('Model snapshot lacks a run record or a checkpoint')
    record = hub.get(f'{prefix}/{record_name}', revision, staging)
    if not matches_local(record, value['files'][record_name]) or json.loads(record.read_text()) != expected_record:
        raise RuntimeError('Remote checkpoint belongs to different code, plans, split or preprocessing; refusing to continue')
    target = staging / value['content_id'] / prefix
    restore_files(hub, revision, prefix, value, target, staging / 'downloads')
    write_json(target / SNAPSHOT, value)
    model_folder.parent.mkdir(parents=True, exist_ok=True)
    if model_folder.exists():
        backup = model_folder.with_name(model_folder.name + f'.before-restore-{time.time_ns()}')
        os.replace(model_folder, backup)
        log.info('Preserved previous incomplete model directory: %s', backup)
    os.replace(target, model_folder)
    log.info('Installed verified checkpoint snapshot from Hugging Face revision %s', revision)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['ensure', 'publish'])
    parser.add_argument('--data', type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    if args.action == 'ensure':
        ensure_prepared(args.data.resolve(), publish=False)
    else:
        publish_prepared(args.data.resolve())


if __name__ == '__main__':
    main()
