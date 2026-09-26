"""FLARE case names, label mapping, count and geometry gates: python tests/test_compare_flare.py."""
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import compare
from cohort_scan import LABEL_IDS

with TemporaryDirectory() as temporary:
    root = Path(temporary)
    source = root / 'source'
    for folder in ('images', 'labels'):
        (source / folder).mkdir(parents=True)
    values = np.arange(27, dtype=np.uint8).reshape(3, 3, 3) % 14
    for i in range(1, 51):
        name = f'FLARE22_Tr_{i:04d}'
        nib.save(nib.Nifti1Image(values, np.eye(4)), source / 'labels' / f'{name}.nii.gz')
        nib.save(nib.Nifti1Image(values, np.eye(4)), source / 'images' / f'{name}_0000.nii.gz')
    with patch.object(compare, 'DATA', root), patch.object(compare, 'snapshot_download', return_value=str(source)):
        raw = compare.fetch_external('flare')
        assert len(list((raw / 'labelsTs').glob('*.nii.gz'))) == 50
        mapped = np.asanyarray(nib.load(raw / 'labelsTs/FLARE22_Tr_0001.nii.gz').dataobj)
        for organ, old in LABEL_IDS['flare'].items():
            assert np.all(mapped[values == old] == compare.LABELS[organ])
        assert (raw / 'imagesTs/FLARE22_Tr_0001_0000.nii.gz').exists()
        (raw / 'done').unlink()
        affine = np.eye(4)
        affine[0, 3] = 5
        nib.save(nib.Nifti1Image(values, affine), source / 'images/FLARE22_Tr_0001_0000.nii.gz')
        try:
            compare.fetch_external('flare')
        except ValueError as error:
            assert 'geometry mismatch' in str(error)
        else:
            raise AssertionError('Geometry mismatch was accepted')
print('FLARE mapping, names, inventory and geometry checks passed')
