"""Offline regression check: python pseudolabels/check_pseudolabel.py."""
from concurrent.futures import ProcessPoolExecutor
from contextlib import redirect_stdout
import io
import json
import multiprocessing
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
from unittest.mock import patch

import nibabel as nib
import numpy as np
import pseudolabel as p
from runtime import resample_img, gated_predict

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tests'))
from test_hf_cache import FakeHub


def gate_probe(gate, active, peak):
    def predict(num_threads_preprocessing=4, num_threads_nifti_save=4):
        with active.get_lock():
            active.value += 1
            peak.value = max(peak.value, active.value)
        time.sleep(0.05)
        with active.get_lock():
            active.value -= 1
    with patch.dict(sys.modules, {'torch': SimpleNamespace(cuda=SimpleNamespace(empty_cache=lambda: None))}):
        gated_predict(predict, gate)()


def check():
    # A nested child process is the regression that the old daemon Pool could not handle.
    with ProcessPoolExecutor(1, mp_context=multiprocessing.get_context('spawn')) as pool:
        assert pool.submit(eval, "__import__('subprocess').check_output([__import__('sys').executable, '-c', 'print(7)']).strip()").result() == b'7'
        assert pool.submit(eval, "__import__('multiprocessing').get_context('spawn').Pool(1).apply(abs, (-7,))").result() == 7

    fixtures = {
        'amos': ['test/imagesTs/unlabelled.nii.gz', 'train/labelsTr/a.nii.gz', 'train/imagesTr/a.nii.gz'],
        'btcv': ['RawData/Training/label/label0001.nii.gz', 'RawData/Training/img/img0001.nii.gz'],
        'flare': ['labels/f.nii.gz', 'images/f_0000.nii.gz'],
    }
    api = SimpleNamespace(list_repo_files=lambda repo, **kw: fixtures[repo])
    with patch.object(p, 'SOURCES', {s: (s, 'pinned', 1) for s in fixtures}):
        assert len(p.scans(api)) == 3
        fixtures['amos'].pop()
        try:
            p.scans(api)
        except RuntimeError:
            pass
        else:
            raise AssertionError('Missing image accepted')

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        source = root / 'ct.nii.gz'
        affine = np.diag([1., 2., 3., 1.])
        nib.save(nib.Nifti1Image(np.zeros((4, 5, 6), np.uint8), affine), source)
        output = root / 'result.nii.gz'
        fake_api = SimpleNamespace(totalsegmentator=lambda ct, *a, **kw: nib.Nifti1Image(np.ones(ct.shape, np.uint8), ct.affine))
        with patch.dict(sys.modules, {'totalsegmentator.python_api': fake_api,
                                     'torch': SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True))}):
            p.label_scan((source, output, np.array([0, 50], np.uint8)))
            result = nib.load(output)
            assert np.all(np.asanyarray(result.dataobj) == 50) and np.array_equal(result.affine, affine)
            # Preserve axis permutation, flips, obliquity and native header forms.
            for affine in (np.diag([-1., 2., -3., 1.]),
                           np.array([[0., -2., 0., 19.], [3., 0., 0., -11.],
                                     [0., 0., 4., 8.], [0., 0., 0., 1.]]),
                           np.array([[1., 0.2, 0., 7.], [0., 2., 0., 8.],
                                     [0., 0., 3., 9.], [0., 0., 0., 1.]])):
                ct = nib.Nifti1Image(np.zeros((4, 5, 6), np.uint8), affine)
                nib.save(ct, source)
                p.label_scan((source, output, np.array([0, 50], np.uint8)))
                result = nib.load(output)
                assert np.allclose(result.affine, affine) and result.shape == ct.shape
                assert int(result.header['sform_code']) == int(ct.header['sform_code'])
            fake_api.totalsegmentator = lambda ct, *a, **kw: nib.Nifti1Image(np.ones(ct.shape, np.uint8), np.eye(4))
            try:
                p.label_scan((source, output, np.array([0, 50], np.uint8)))
            except RuntimeError:
                pass
            else:
                raise AssertionError('Wrong grid accepted')

        cases = {f'amos/{i}.nii.gz': ('amos', f'{i}.nii.gz') for i in range(3)}
        hub = FakeHub()
        hub.api = None
        spec = {'versions': {'TotalSegmentator': 'test'}}
        calls = []
        def predict(job):
            calls.append(job[0].name)
            job[1].parent.mkdir(parents=True, exist_ok=True)
            job[1].write_bytes(b'prediction')
            return job[1]
        with patch.object(p, 'scans', return_value=cases), patch.object(p, 'lookup', return_value=None), \
                patch.object(p, 'snapshot_download', return_value=tmp), patch.object(p, 'label_scan', side_effect=predict):
            hub.fail_commit = 2
            try:
                p.run(root, 1, hub, spec)
            except ConnectionError:
                pass
            else:
                raise AssertionError('Upload failure swallowed')
            manifest_name = next(n for n in hub.revisions[hub.head] if n.endswith('dataset.json'))
            manifest = json.loads(hub.revisions[hub.head][manifest_name])
            assert not manifest['complete'] and len(manifest['files']) == 1
            hub.fail_commit = None
            calls.clear()
            p.run(root, 1, hub, spec)
            assert calls == ['1.nii.gz', '2.nii.gz']
            calls.clear()
            p.run(root, 1, hub, spec)
            assert not calls  # Completed resume does not run inference or download CTs.
            remote = hub.revisions[hub.head]
            remote[next(n for n in remote if n.endswith('.nii.gz'))] = b'corrupt'
            try:
                p.run(root, 1, hub, spec)
            except RuntimeError:
                pass
            else:
                raise AssertionError('Corrupt remote output accepted')
            empty = FakeHub()
            empty.api = None
            with patch.object(p, 'label_scan', side_effect=RuntimeError('GPU failed')):
                try:
                    p.run(root, 1, empty, spec)
                except RuntimeError:
                    pass
                else:
                    raise AssertionError('Inference failure swallowed')
            assert not empty.commits

    # Compare interpolation against the actual endpoint convention used by TS.
    from scipy.ndimage import zoom
    rng = np.random.default_rng(42)
    for shape, target in [((7, 8, 9), (11, 5, 13)), ((1, 5, 7), (3, 1, 9)),
                          ((8, 9, 10), (3, 4, 5)), ((4, 5, 6), (4, 5, 6))]:
        factors = np.array(target) / shape
        ct = rng.normal(size=shape)
        np.testing.assert_allclose(resample_img(ct, factors, order=1),
                                   zoom(ct, factors, order=1, mode='nearest'), atol=1e-12)
        labels = rng.integers(0, 67, size=shape, dtype=np.uint8)
        result = resample_img(labels, factors, order=0)
        np.testing.assert_array_equal(result, zoom(labels, factors, order=0, mode='nearest'))
        assert set(np.unique(result)) <= set(np.unique(labels))

    # A deliberately permuted teacher ID map must still map by anatomy name.
    teacher = {i: name for i, name in enumerate(reversed([*p.D998, *p.MERGED]), 1)}
    with patch.dict(sys.modules, {'totalsegmentator.map_to_binary': SimpleNamespace(class_map={'total': teacher})}):
        lut = p.lookup()
        for source_id, name in teacher.items():
            assert lut[source_id] == p.D998.index(p.MERGED.get(name, name)) + 1
    assert sum(source[2] for source in p.SOURCES.values()) == 380
    assert set(p.SOURCES) == {'amos', 'btcv', 'flare'}

    # Nested process counts are capped and the GPU gate releases on failure.
    gate = multiprocessing.get_context('spawn').BoundedSemaphore(1)
    def fail(num_threads_preprocessing=4, num_threads_nifti_save=4):
        assert num_threads_preprocessing == num_threads_nifti_save == 1
        assert not gate.acquire(block=False)
        raise RuntimeError('prediction failed')
    cleanup = []
    with patch.dict(sys.modules, {'torch': SimpleNamespace(cuda=SimpleNamespace(empty_cache=lambda: cleanup.append(True)))}):
        try:
            gated_predict(fail, gate)()
        except RuntimeError:
            pass
        else:
            raise AssertionError('Prediction failure swallowed by GPU gate')
    assert cleanup and gate.acquire(block=False)
    gate.release()
    context = multiprocessing.get_context('spawn')
    active, peak = context.Value('i', 0), context.Value('i', 0)
    probes = [context.Process(target=gate_probe, args=(gate, active, peak)) for _ in range(3)]
    for process in probes:
        process.start()
    for process in probes:
        process.join(30)
        if process.is_alive():
            process.terminate()
            process.join()
            raise AssertionError('GPU gate deadlocked')
        assert process.exitcode == 0
    assert peak.value == 1 and active.value == 0


if __name__ == '__main__':
    with redirect_stdout(io.StringIO()):
        check()
    print('=== pseudo-label offline checks passed', flush=True)
