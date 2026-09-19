"""Offline integrity/recovery tests. No network, GPU or clinical data required.

    python -m unittest discover -s tests -p test_hf_cache.py -v
"""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import hf_cache as cache


class FakeHub:
    def __init__(self):
        self.revisions = {'r0': {'.gitattributes': cache.CACHE_ATTRIBUTES.encode()}}
        self.head = 'r0'
        self.commits = []
        self.gets = []
        self.fail_commit = None
        self.corrupt_get = False

    def info(self):
        entries = {}
        for name, value in self.revisions[self.head].items():
            # Exercise both Git and LFS integrity metadata.
            lfs = SimpleNamespace(sha256=hashlib.sha256(value).hexdigest()) if name.endswith(('.b2nd', '.pth', '.tar')) else None
            entries[name] = SimpleNamespace(size=len(value), lfs=lfs,
                blob_id=hashlib.sha1(f'blob {len(value)}\0'.encode() + value).hexdigest())
        return self.head, entries

    def get(self, name, revision, directory, force=False):
        self.gets.append((name, revision, force))
        value = self.revisions[revision][name]
        if self.corrupt_get and name.endswith('.pth'):
            value = b'bad download'
        out = directory / name
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(value)
        return out

    def commit(self, additions, message, parent):
        if self.fail_commit == len(self.commits) + 1:
            raise ConnectionError('simulated upload interruption')
        if parent != self.head:
            raise RuntimeError('parent conflict')
        files = dict(self.revisions[self.head])
        files.update({n: v.read_bytes() if isinstance(v, Path) else v for n, v in additions.items()})
        self.commits.append((additions, message))
        self.head = f'r{len(self.commits)}'
        self.revisions[self.head] = files
        return self.head


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.data = self.root / 'data'
        self.data.mkdir()
        self.hub = FakeHub()

    def tearDown(self):
        self.tmp.cleanup()

    def file(self, relative, value=b'example'):
        out = self.data / relative
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(value)
        return out

    def small_manifest(self, count=3):
        files = {}
        for i in range(count):
            name = f'nnUNet_preprocessed/{cache.DATASET}/case_{i}.b2nd'
            files[name] = cache.file_info(self.file(name, f'case {i}'.encode()))
        return cache.manifest('preprocessed', 'recipe', files)

    def bundle(self):
        pre = f'nnUNet_preprocessed/{cache.DATASET}'
        raw = f'nnUNet_raw/{cache.DATASET}'
        config = pre + '/AtlasNetPlans_3d_fullres'
        def js(name, value):
            self.file(name, json.dumps(value).encode())
        dataset = {'numTraining': 2}
        js(pre + '/dataset.json', dataset)
        js(raw + '/dataset.json', dataset)
        js(pre + '/AtlasNetPlans.json', {'configurations': {'3d_fullres': {'data_identifier': 'AtlasNetPlans_3d_fullres'}}})
        js(pre + '/splits_final.json', [{'train': ['a'], 'val': ['b']}] * 5)
        self.file(raw + '/build_dataset.csv', b'name,role\na,train\nb,train\nc,test\n')
        js(raw + '/build_dataset.json', {'counts': {'train': 2, 'test': 1, 'failed': 0}})
        js(config + '/fg_sampling/meta.json', {'identifiers': ['a', 'b']})
        for name in ('locations.b2nd', 'indptr.npy', 'class_id.npy', 'count.npy', 'base.npy', 'shape.npy'):
            self.file(config + '/fg_sampling/' + name)
        for case in ('a', 'b'):
            for suffix in ('.b2nd', '_seg.b2nd', '.pkl'):
                self.file(config + '/' + case + suffix)
            self.file(pre + '/gt_segmentations/' + case + '.nii.gz')
        self.file(raw + '/imagesTs/c_0000.nii.gz')
        self.file(raw + '/labelsTs/c.nii.gz')
        atlas = self.file('atlas/checkpoint_final.pth')
        return atlas, pre, raw, config

    def model(self):
        folder = self.root / 'results' / 'AdrenalMultiorgan__AtlasNetPlans__3d_fullres'
        (folder / 'fold_0').mkdir(parents=True)
        record = {'trainer': 'same', 'preprocessed_content_id': 'cache123'}
        cache.write_json(folder / 'fold_0/run.json', record)
        (folder / 'fold_0/checkpoint_latest.pth').write_bytes(b'full checkpoint with optimizer and epoch')
        cache.write_json(folder / 'plans.json', {'plans': 'same'})
        cache.write_json(folder / 'dataset.json', {'data': 'same'})
        return folder, record

    def test_binary_tracking_rules_are_added_without_erasing_existing_rules(self):
        self.hub.revisions['r0']['.gitattributes'] = b'*.zip filter=lfs diff=lfs merge=lfs -text\n'
        cache.publish_cache(self.hub, self.data, self.small_manifest())
        attributes = self.hub.revisions[self.hub.head]['.gitattributes'].decode()
        self.assertIn('*.zip filter=lfs', attributes)
        self.assertIn('preprocessed/**/*.tar filter=lfs', attributes)
        self.assertEqual(list(self.hub.commits[0][0]), ['.gitattributes'])

    def test_ensure_first_build_is_published_before_ready_marker(self):
        atlas, _, _, _ = self.bundle()
        with patch.object(cache, 'recipe_id', return_value='recipe'), \
                patch.object(cache, 'ATLASNET_SHA256', cache.file_info(atlas)['sha256']), \
                patch.object(cache.subprocess, 'run') as run:
            cache.ensure_prepared(self.data, self.root, self.hub)
            run.assert_called_once_with(['bash', str(self.root / 'prepare.sh'), str(self.data)],
                                        cwd=self.root, check=True)
        self.assertIn(cache.complete_path('recipe'), self.hub.revisions[self.hub.head])
        self.assertTrue((self.data / '.prepared').exists())
        self.assertTrue((self.data / '.preprocessed-local.json').exists())

    def test_recipe_tracks_code_but_not_logs(self):
        for name in ('cohort_scan.py', 'build_dataset.py', 'preprocess.py', 'prepare.sh',
                     'atlasnet_plans.json', 'pixi.lock', 'vendor/nnUNet/example.py'):
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('original')
        original = cache.recipe_id(self.root)
        (self.root / 'run.log').write_text('unrelated')
        self.assertEqual(cache.recipe_id(self.root), original)
        (self.root / 'vendor/nnUNet/example.py').write_text('changed')
        self.assertNotEqual(cache.recipe_id(self.root), original)

    def test_manifest_tampering_rejected(self):
        value = self.small_manifest()
        value['files'][next(iter(value['files']))]['size'] += 1
        with self.assertRaisesRegex(RuntimeError, 'checksum'):
            cache.check_manifest(value, 'preprocessed')

    def test_recipe_mismatch_rejected(self):
        with self.assertRaisesRegex(RuntimeError, 'recipe'):
            cache.check_manifest(self.small_manifest(), 'preprocessed', 'different')

    def test_traversal_and_symlink_rejected(self):
        for name in ('../outside', '/absolute', 'a/../../b', 'a\\b', 'a//b', 'C:/secret'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                cache.safe_path(self.data, name)
        (self.data / 'escape').symlink_to(self.root, target_is_directory=True)
        with self.assertRaises(ValueError):
            cache.safe_path(self.data, 'escape/secret')

    def test_complete_marker_is_last_and_upload_is_idempotent(self):
        value = self.small_manifest()
        cache.publish_cache(self.hub, self.data, value)
        self.assertEqual(list(self.hub.commits[-1][0]), [cache.complete_path('recipe')])
        commits = len(self.hub.commits)
        cache.publish_cache(self.hub, self.data, value)
        self.assertEqual(len(self.hub.commits), commits)

    def test_interrupted_upload_never_publishes_completion_and_resumes_batches(self):
        value = self.small_manifest(70)
        self.hub.fail_commit = 2
        with self.assertRaises(ConnectionError):
            cache.publish_cache(self.hub, self.data, value)
        self.assertNotIn(cache.complete_path('recipe'), self.hub.revisions[self.hub.head])
        self.assertTrue(next(iter(self.hub.commits[0][0])).endswith('.tar'))
        self.hub.fail_commit = None
        cache.publish_cache(self.hub, self.data, value)
        self.assertEqual(list(self.hub.commits[1][0]), [cache.complete_path('recipe')])
        self.assertIn(cache.complete_path('recipe'), self.hub.revisions[self.hub.head])

    def test_remote_payload_corruption_rejected(self):
        value = self.small_manifest()
        cache.publish_cache(self.hub, self.data, value)
        name = 'part-00000.tar'
        self.hub.revisions[self.hub.head][cache.cache_prefix(value) + '/' + name] = b'wrong'
        with self.assertRaisesRegex(RuntimeError, 'corrupt'):
            cache.publish_cache(self.hub, self.data, value)

    def test_completed_cache_not_overwritten(self):
        value = self.small_manifest()
        cache.publish_cache(self.hub, self.data, value)
        other = self.small_manifest(4)
        with self.assertRaisesRegex(RuntimeError, 'overwrite'):
            cache.publish_cache(self.hub, self.data, other)

    def test_tar_download_pinned_and_files_verified(self):
        value = self.small_manifest()
        cache.publish_cache(self.hub, self.data, value)
        revision = self.hub.head
        target = self.root / 'restored'
        value = json.loads(self.hub.revisions[revision][cache.complete_path('recipe')])
        cache.restore_cache(self.hub, revision, value, target)
        cache.verify_local(target, value)
        self.assertFalse(list((target / '.hf-downloads').rglob('*.tar')))
        self.assertTrue(all(r == revision for n, r, _ in self.hub.gets if n != '.gitattributes'))

    def test_missing_case_prevents_cache(self):
        _, _, _, config = self.bundle()
        (self.data / config / 'a_seg.b2nd').unlink()
        with self.assertRaisesRegex(RuntimeError, 'Incomplete'):
            cache.cache_files(self.data)

    def test_missing_test_scan_prevents_cache(self):
        _, _, raw, _ = self.bundle()
        (self.data / raw / 'imagesTs/c_0000.nii.gz').unlink()
        with self.assertRaisesRegex(RuntimeError, 'Incomplete'):
            cache.cache_files(self.data)

    def test_incomplete_foreground_index_prevents_cache(self):
        _, _, _, config = self.bundle()
        cache.write_json(self.data / config / 'fg_sampling/meta.json', {'identifiers': ['a']})
        with self.assertRaisesRegex(RuntimeError, 'Foreground'):
            cache.cache_files(self.data)

    def test_split_overlap_prevents_cache(self):
        _, pre, _, _ = self.bundle()
        cache.write_json(self.data / pre / 'splits_final.json', [{'train': ['a', 'b'], 'val': ['b']}] * 5)
        with self.assertRaisesRegex(RuntimeError, 'overlapping'):
            cache.cache_files(self.data)

    def test_complete_bundle_excludes_raw_training_and_logs(self):
        atlas, pre, raw, _ = self.bundle()
        self.file(raw + '/imagesTr/a_0000.nii.gz', b'not needed')
        self.file(pre + '/preprocess-console.log', b'not payload')
        with patch.object(cache, 'ATLASNET_SHA256', cache.file_info(atlas)['sha256']):
            value = cache.build_cache_manifest(self.data, 'recipe')
        self.assertNotIn(raw + '/imagesTr/a_0000.nii.gz', value['files'])
        self.assertNotIn(pre + '/preprocess-console.log', value['files'])
        self.assertIn('atlas/checkpoint_final.pth', value['files'])
        cache.check_manifest(value, 'preprocessed', 'recipe')

    def test_ensure_fresh_vm_restores_without_preparation(self):
        atlas, _, _, _ = self.bundle()
        with patch.object(cache, 'ATLASNET_SHA256', cache.file_info(atlas)['sha256']):
            value = cache.build_cache_manifest(self.data, 'recipe')
        cache.publish_cache(self.hub, self.data, value)
        fresh = self.root / 'fresh'
        with patch.object(cache, 'recipe_id', return_value='recipe'), patch.object(cache.subprocess, 'run') as run:
            cache.ensure_prepared(fresh, self.root, self.hub)
            run.assert_not_called()
        self.assertEqual(json.loads((fresh / '.prepared').read_text())['content_id'], value['content_id'])
        cache.verify_local(fresh, value)

    def test_ensure_upload_retry_does_not_rebuild(self):
        atlas, _, _, _ = self.bundle()
        with patch.object(cache, 'ATLASNET_SHA256', cache.file_info(atlas)['sha256']):
            value = cache.build_cache_manifest(self.data, 'recipe')
        cache.write_json(self.data / '.preprocessed-local.json', value)
        with patch.object(cache, 'recipe_id', return_value='recipe'), patch.object(cache.subprocess, 'run') as run:
            cache.ensure_prepared(self.data, self.root, self.hub)
            run.assert_not_called()
        self.assertTrue((self.data / '.prepared').is_file())

    def test_ensure_permission_error_is_not_cache_miss(self):
        with patch.object(cache, 'recipe_id', return_value='recipe'), patch.object(self.hub, 'info', side_effect=PermissionError), \
                patch.object(cache.subprocess, 'run') as run:
            with self.assertRaises(PermissionError):
                cache.ensure_prepared(self.data, self.root, self.hub)
            run.assert_not_called()
        self.assertFalse((self.data / '.prepared').exists())

    def test_model_is_one_commit_and_restores_transactionally(self):
        folder, record = self.model()
        cache.publish_model(folder, 'epoch 100', self.hub)
        self.assertEqual(len(self.hub.commits), 1)
        self.assertIn(folder.name + '/' + cache.SNAPSHOT, self.hub.commits[0][0])
        fresh = self.root / 'fresh-model' / folder.name
        self.assertTrue(cache.restore_model(fresh, record, self.hub))
        self.assertEqual((fresh / 'fold_0/checkpoint_latest.pth').read_bytes(),
                         (folder / 'fold_0/checkpoint_latest.pth').read_bytes())

    def test_model_record_mismatch_never_installs_checkpoint(self):
        folder, _ = self.model()
        cache.publish_model(folder, 'epoch 100', self.hub)
        fresh = self.root / 'fresh-model' / folder.name
        with self.assertRaisesRegex(RuntimeError, 'different code'):
            cache.restore_model(fresh, {'trainer': 'different'}, self.hub)
        self.assertFalse(fresh.exists())

    def test_model_corrupt_download_never_installs_checkpoint(self):
        folder, record = self.model()
        cache.publish_model(folder, 'epoch 100', self.hub)
        self.hub.corrupt_get = True
        fresh = self.root / 'fresh-model' / folder.name
        with self.assertRaisesRegex(RuntimeError, 'checksum'):
            cache.restore_model(fresh, record, self.hub)
        self.assertFalse(fresh.exists())

    def test_missing_model_manifest_never_starts_over(self):
        folder, record = self.model()
        self.hub.commit({folder.name + '/fold_0/checkpoint_latest.pth': b'old checkpoint'}, 'legacy', self.hub.head)
        with self.assertRaisesRegex(RuntimeError, 'refusing to start over'):
            cache.restore_model(self.root / 'fresh' / folder.name, record, self.hub)

    def test_no_remote_model_allows_new_run(self):
        folder, record = self.model()
        self.assertFalse(cache.restore_model(self.root / 'fresh' / folder.name, record, self.hub))

    def test_model_ignores_tmp_and_stale_remote_files(self):
        folder, record = self.model()
        (folder / 'fold_0/checkpoint_bad.pth.tmp').write_bytes(b'partial')
        cache.publish_model(folder, 'epoch 100', self.hub)
        self.hub.commit({folder.name + '/fold_0/checkpoint_final.pth': b'stale'}, 'stale file', self.hub.head)
        fresh = self.root / 'fresh-model' / folder.name
        cache.restore_model(fresh, record, self.hub)
        self.assertFalse((fresh / 'fold_0/checkpoint_final.pth').exists())
        self.assertFalse((fresh / 'fold_0/checkpoint_bad.pth.tmp').exists())

    def test_failed_model_commit_preserves_last_snapshot(self):
        folder, _ = self.model()
        cache.publish_model(folder, 'epoch 100', self.hub)
        old_head = self.hub.head
        old_state = copy.deepcopy(self.hub.revisions[old_head])
        (folder / 'fold_0/checkpoint_latest.pth').write_bytes(b'epoch 200')
        self.hub.fail_commit = 2
        with self.assertRaises(ConnectionError):
            cache.publish_model(folder, 'epoch 200', self.hub)
        self.assertEqual(self.hub.head, old_head)
        self.assertEqual(self.hub.revisions[self.hub.head], old_state)


if __name__ == '__main__':
    unittest.main()
