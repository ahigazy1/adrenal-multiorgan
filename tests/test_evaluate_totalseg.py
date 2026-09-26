import gzip
import json
import sys
from pathlib import Path
import tempfile
import unittest
import zipfile
from unittest.mock import patch

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cohort_scan import sha256
from evaluate import volume_metrics
from evaluate_totalseg import memberships, score


class EvaluationTests(unittest.TestCase):
    def test_absent_reference_is_not_a_zero_percent_error(self):
        empty = volume_metrics(0, 0, 0, 0.002)
        self.assertIsNone(empty['dice'])
        false_positive = volume_metrics(0, 10, 0, 0.002)
        self.assertEqual(false_positive['dice'], 0)
        self.assertEqual(false_positive['volume_error_ml'], 0.02)
        self.assertIsNone(false_positive['volume_error_pct'])
        missing = volume_metrics(10, 0, 0, 0.002)
        self.assertEqual(missing['volume_error_pct'], -100)
        self.assertTrue(missing['miss'])

    def test_splits_must_match_build(self):
        with tempfile.TemporaryDirectory() as folder:
            build, splits = Path(folder) / 'build.csv', Path(folder) / 'splits.json'
            build.write_text('name,role\nts_a,train\nts_b,train\nts_c,test\n')
            splits.write_text(json.dumps([{'train': ['ts_a'], 'val': ['ts_b']}]))
            self.assertEqual(memberships(build, splits), {'ts_a': 'train', 'ts_b': 'validation', 'ts_c': 'test'})
            splits.write_text(json.dumps([{'train': ['ts_a'], 'val': ['ts_a']}]))
            with self.assertRaises(ValueError):
                memberships(build, splits)

    def test_raw_union_volume_grid_and_hash(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            out = root / 'worker-0/native_labels'
            out.mkdir(parents=True)
            affine = np.diag([2., 3., 4., 1.])
            pred = np.zeros((3, 3, 3), dtype=np.uint8)
            pred[0, 0, 0] = 3
            mask = out / 'ts_s0001.nii.gz'
            nib.save(nib.Nifti1Image(pred, affine), mask)
            record = {'case': 'ts_s0001', 'sha256': sha256(mask)}
            archive = root / 'raw.zip'
            with zipfile.ZipFile(archive, 'w') as z:
                for structure, voxel in [('kidney_left', (0, 0, 0)), ('kidney_cyst_left', (1, 0, 0))]:
                    truth = np.zeros_like(pred)
                    truth[voxel] = 1
                    z.writestr(f's0001/segmentations/{structure}.nii.gz',
                               gzip.compress(nib.Nifti1Image(truth, affine).to_bytes()))
            with patch('evaluate_totalseg.TS_FILES', {'kidney_left': ['kidney_left', 'kidney_cyst_left']}):
                row = score(record, root, archive, {})[0]
                self.assertAlmostEqual(row['reference_ml'], 0.048)
                self.assertAlmostEqual(row['dice'], 2 / 3)
                self.assertEqual(row['volume_error_pct'], -50)
                self.assertEqual(row['split'], 'excluded_from_dataset902')
                moved = affine.copy(); moved[0, 3] = 10
                nib.save(nib.Nifti1Image(pred, moved), mask)
                with self.assertRaisesRegex(ValueError, 'changed prediction'):
                    score(record, root, archive, {})
                record['sha256'] = sha256(mask)
                with self.assertRaisesRegex(ValueError, 'grid mismatch'):
                    score(record, root, archive, {})
            from cohort_scan import _open_zips
            _open_zips.pop(archive).close()


if __name__ == '__main__':
    unittest.main()
