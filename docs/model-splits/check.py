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
assert len(references) == 56
assert json.loads((root / 'sources/D998-dataset.json').read_text())['numTraining'] == 1128
assert set(d997['val']) - set(d998['val']) == {'s0352'}
assert not set(d998['val']) - set(d997['val'])

# D998's training IDs: its dataset_fingerprint.json lists every case's shape after nnU-Net's crop-to-nonzero, in
# nnU-Net's sorted case order. Align that sequence in order to the TotalSegmentator v2.0.1 CT shapes (axes reversed by
# the reorienting reader); a crop can only shrink a shape. The least-cropping alignment must be unique.
shapes_after_crop = [tuple(s) for s in json.loads((root / 'sources/D998-dataset_fingerprint.json').read_text())['shapes_after_crop']]
ts = json.loads((root / 'sources/totalsegmentator-v201-ct-shapes.json').read_text())
names = sorted(ts)
ct = [tuple(reversed(ts[n][0])) for n in names]
N, M, INF = len(shapes_after_crop), len(names), float('inf')
cost = [[sum(b - a for a, b in zip(s, c)) if all(a <= b for a, b in zip(s, c)) else INF for c in ct] for s in shapes_after_crop]
after = [[INF] * (M + 1) for _ in range(N)] + [[0] * (M + 1)]  # least cost aligning cases i.. to names j..
for i in range(N - 1, -1, -1):
    for j in range(M - 1, -1, -1):
        after[i][j] = min(after[i][j + 1], cost[i][j] + after[i + 1][j + 1])
before = [[0] * (M + 1)] + [[INF] * (M + 1) for _ in range(N)]  # least cost aligning cases ..i to names ..j
for i in range(1, N + 1):
    for j in range(1, M + 1):
        before[i][j] = min(before[i][j - 1], before[i - 1][j - 1] + cost[i - 1][j - 1])
best = after[0][0]
matched = []
for i in range(N):
    options = [j for j in range(M) if before[i][j] + cost[i][j] + after[i + 1][j + 1] == best]
    assert len(options) == 1, f'case {i} of the fingerprint has {len(options)} equally good matches'
    matched.append(names[options[0]])
assert N == 1128 and max(cost[i][names.index(n)] for i, n in enumerate(matched)) <= 2
assert set(d998['val']) <= set(matched), 'the recovered validation cases must all be in the reconstructed pool'
assert d998['train'] == sorted(set(matched) - set(d998['val'])) and len(d998['train']) == 1072
assert set(d998['train']) <= set(d997['train']) and not set(d998['train']) & set(d997['val'])
print('Split evidence verified: D997 1082 corroborated train / 57 verified validation; '
      'D998 1072 train reconstructed uniquely from its fingerprint / 56 verified validation.')
