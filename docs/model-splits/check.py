"""Verify the committed recovery evidence: python docs/model-splits/check.py."""
import hashlib
import json
from pathlib import Path
import re

root = Path(__file__).resolve().parent
provenance = json.loads((root / 'provenance.json').read_text())
for name, record in provenance['sources'].items():
    assert hashlib.sha256((root / 'sources' / name).read_bytes()).hexdigest() == record['sha256'], name
d997 = json.loads((root / 'D997.json').read_text())
d998 = json.loads((root / 'D998.json').read_text())
for record in (d997, d998):
    for key in ('train', 'val'):
        values = record[key]
        if values is not None:
            assert len(values) == len(set(values))
            assert all(re.fullmatch(r's\d{4}', case) for case in values)
    assert record['fold'] == 0 and record['test'] is None
assert len(d997['train']) == 1082 and len(d997['val']) == 57
assert not set(d997['train']) & set(d997['val'])
for field, filename in [('train', 'D997-split0_training_samples.txt'), ('val', 'D997-split0_validation_samples.txt')]:
    assert d997[field] == (root / 'sources' / filename).read_text().splitlines()
log = (root / 'sources/D997-final-training-log.txt').read_text()
assert 'This split has 1082 training and 57 validation cases.' in log
assert set(re.findall(r'predicting (s\d+)', log)) == set(d997['val'])
assert json.loads((root / 'sources/D997-dataset.json').read_text())['numTraining'] == 1139
summary = json.loads((root / 'sources/D998-validation-summary.json').read_text())
references = [Path(row['reference_file']).name.removesuffix('.nii.gz') for row in summary['metric_per_case']]
predictions = [Path(row['prediction_file']).name.removesuffix('.nii.gz') for row in summary['metric_per_case']]
assert references == predictions == d998['val']
assert len(references) == 56 and d998['train'] is None
assert json.loads((root / 'sources/D998-dataset.json').read_text())['numTraining'] == 1128
assert d998['counts']['train_inferred_if_validation_summary_covers_full_fold'] == 1128 - len(references)
assert set(d997['val']) - set(d998['val']) == {'s0352'}
assert not set(d998['val']) - set(d997['val'])
print('Split evidence verified: D997 1082 corroborated train / 57 verified validation; D998 56 verified validation, training IDs unknown.')
