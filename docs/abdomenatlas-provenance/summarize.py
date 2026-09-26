"""Verify recovered exports and rebuild the portable case mapping and counts (stdlib only)."""
import csv
from collections import Counter
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
manifest = json.loads((root / 'provenance.json').read_text())
for name, record in manifest['source_files'].items():
    assert hashlib.sha256((root / 'sources' / name).read_bytes()).hexdigest() == record['sha256'], name

def read(name):
    with (root / 'sources' / name).open(newline='', encoding='utf-8-sig') as file:
        return list(csv.DictReader(file))

first = read('atlas_overlap_shards01_05.csv')
later = read('atlas_overlap_export_up_to_shard13.csv')
early = read('atlas_overlap_shards06_10.csv')
shard1 = read('atlas_overlap_shard01.csv')
assert all(row in later for row in early), 'Older export conflicts with newer export'
assert all(dict(shard='shard01', **row) in first for row in shard1), 'Shard 1 snapshot conflicts'
rows = first + later
assert len(rows) == 279
assert len({row['atlas_bdmap_id'] for row in rows}) == len(rows)
assert len({(row['gold_dataset'], row['gold_scan']) for row in rows}) == len(rows)
assert set(row['shard'] for row in later) == {f'shard{i:02d}' for i in range(6, 12)}
status = {'exact': 'standardized_image_hash_match', 'shape+corr': 'shape_and_correlation_candidate',
          'corr-only': 'correlation_only_candidate'}
for row in rows:
    dataset, scan = row['gold_dataset'], row['gold_scan'].removesuffix('.nii.gz')
    if dataset == 'BTCV':
        assert scan.startswith('img')
        case = 'btcv_label' + scan[3:]
    elif dataset == 'FLARE':
        assert scan.endswith('_0000')
        case = scan.removesuffix('_0000')
    else:
        assert dataset == 'AMOS' and scan.startswith('amos_')
        case = scan
    row['evaluation_case_id'] = case
    row['evidence_status'] = status[row['match_type']]
    row['source_export'] = 'atlas_overlap_shards01_05.csv' if row in first else 'atlas_overlap_export_up_to_shard13.csv'
    assert .97 <= float(row['corr32']) <= 1
rows.sort(key=lambda row: row['atlas_bdmap_id'])
with (root / 'case_mapping.csv').open('w', newline='', encoding='utf-8') as file:
    writer = csv.DictWriter(file, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
counts = Counter((row['gold_dataset'], row['match_type']) for row in rows)
summary = {
    'atlas_collection': manifest['atlas_dataset_named_in_scripts'],
    'total_recorded_matches': len(rows),
    'by_dataset': {dataset: {method: counts[dataset, method] for method in status} for dataset in ('AMOS', 'BTCV', 'FLARE')},
    'by_method': dict(Counter(row['match_type'] for row in rows)),
    'by_shard_with_positive_rows': dict(sorted(Counter(row['shard'] for row in rows).items())),
    'correlation_only_candidates': [row for row in rows if row['match_type'] == 'corr-only'],
    'coverage_status': 'incomplete; no complete processed-shard inventory or negative-match manifest recovered',
    'not_established': ['complete AbdomenAtlas source attribution', 'AtlasNet checkpoint training/validation membership',
                        'AbdomenAtlas 3.0 cross-version identity', 'independence of scans without a recorded match'],
}
(root / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
assert summary['by_method'] == {'exact': 111, 'shape+corr': 167, 'corr-only': 1}
print('Verified 279 unique mappings: AMOS199, BTCV30, FLARE50; 111 standardized hash matches, 168 correlation candidates.')
