"""Offline monitor integrity and restart checks; standard library only."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import monitor as m


def fixture(counts):
    recipe = {'sources': {name: ['repo', 'revision', total] for name, total in m.EXPECTED.items()},
              'labels': [f'label_{i}' for i in range(1, 67)],
              'versions': {'TotalSegmentator': '2.18.0'},
              'code': 'runner', 'runtime_code': 'runtime', 'requirements': 'requirements'}
    prefix = f'pseudolabels/totalseg-2.18.0/{m.digest(recipe)[:16]}'
    files = {f'{name}/{i}.nii.gz': {'size': 10, 'sha256': 'a' * 64, 'git_sha1': 'b' * 40}
             for name, count in counts.items() for i in range(count)}
    manifest = {'recipe': recipe, 'scans_total': 380, 'files': files, 'complete': counts == m.EXPECTED}
    entries = {f'{prefix}/{name}': SimpleNamespace(size=10, lfs={'sha256': 'a' * 64}, blob_id=None)
               for name in files}
    return manifest, entries, prefix


class MonitorChecks(unittest.TestCase):
    def test_partial_and_complete(self):
        manifest, entries, prefix = fixture({'amos': 26})
        result = m.progress(manifest, entries, prefix)
        self.assertEqual(result['verified'], 26)
        self.assertEqual(result['state'], 'IN_PROGRESS')
        self.assertEqual(result['sources']['btcv']['verified'], 0)
        self.assertEqual(m.progress(*fixture(m.EXPECTED))['state'], 'COMPLETE')

    def test_corruption_is_not_progress(self):
        manifest, entries, prefix = fixture({'amos': 1})
        next(iter(entries.values())).lfs['sha256'] = 'c' * 64
        with self.assertRaises(RuntimeError):
            m.progress(manifest, entries, prefix)

    def test_false_completion_and_old_cohort(self):
        manifest, entries, prefix = fixture({'amos': 1})
        manifest['complete'] = True
        with self.assertRaises(RuntimeError):
            m.progress(manifest, entries, prefix)
        manifest['scans_total'] = 793
        with self.assertRaises(RuntimeError):
            m.progress(manifest, entries, prefix)

    def test_monitor_restart_pins_run_without_credentials(self):
        hashes = {'code': {'runner'}, 'runtime_code': {'runtime'}, 'requirements': {'requirements'}}
        manifest, entries, prefix = fixture({'amos': 26})
        snapshot = {'checked_at': 'now', 'hf': m.progress(manifest, entries, prefix)}
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / 'state.json'
            self.assertIsNone(m.load_state(path, 'repo', 'project', 'instance', hashes))
            m.save_state(path, 'repo', 'project', 'instance', hashes, snapshot)
            self.assertEqual(m.load_state(path, 'repo', 'project', 'instance', hashes), prefix)
            text = path.read_text()
            self.assertNotIn('token', text.lower())
            self.assertEqual(json.loads(text)['last_verified'], 26)
            self.assertFalse(path.with_name('state.json.tmp').exists())
            changed = {**hashes, 'code': {'new-code'}}
            with self.assertRaises(RuntimeError):
                m.load_state(path, 'repo', 'project', 'instance', changed)
            with self.assertRaises(RuntimeError):
                m.load_state(path, 'other-repo', 'project', 'instance', hashes)

    def test_reconnect_uses_remote_manifest_not_saved_count(self):
        manifest, entries, prefix = fixture({'amos': 51})
        info = {'private': True, 'sha': 'revision', 'siblings': [
            {'rfilename': name, 'size': entry.size, 'lfs': entry.lfs} for name, entry in entries.items()]
            + [{'rfilename': prefix + '/dataset.json'}]}
        hashes = {'code': {'runner'}, 'runtime_code': {'runtime'}, 'requirements': {'requirements'}}
        with patch.object(m, 'get_json', side_effect=[info, manifest]):
            result = m.hf_snapshot('not-a-real-token', 'owner/repo', prefix, hashes)
        self.assertEqual(result['verified'], 51)
        self.assertTrue(result['matches_local_code'])
        # A different local recipe never adopts old progress automatically.
        with patch.object(m, 'get_json', side_effect=[info, manifest]):
            result = m.hf_snapshot('not-a-real-token', 'owner/repo', None, {**hashes, 'code': {'new-code'}})
        self.assertEqual(result['state'], 'WAITING_FOR_MANIFEST')

    def test_repository_can_be_made_private_without_changing_run(self):
        manifest, entries, prefix = fixture({'amos': 1})
        info = {'sha': 'revision', 'siblings': [
            {'rfilename': name, 'size': entry.size, 'lfs': entry.lfs} for name, entry in entries.items()]
            + [{'rfilename': prefix + '/dataset.json'}]}
        hashes = {'code': {'runner'}, 'runtime_code': {'runtime'}, 'requirements': {'requirements'}}
        for private in (False, True):
            with patch.object(m, 'get_json', side_effect=[{**info, 'private': private}, manifest]):
                result = m.hf_snapshot('not-a-real-token', 'owner/repo', prefix, hashes)
            self.assertEqual(result['verified'], 1)
            self.assertEqual(result['prefix'], prefix)
            self.assertEqual(result['visibility'], 'private' if private else 'public')


if __name__ == '__main__':
    unittest.main()
