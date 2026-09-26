# AbdomenAtlas source-dataset overlap recovered from AdrenalGold

The historical image-matching work was found in `C:/Users/ahiga/Desktop/AdrenalGold`, with later exports in Downloads. It maps **AbdomenAtlas 1.1 Mini BDMAP IDs to matching or candidate AMOS, BTCV and FLARE scans**. This is recovered image-overlap evidence, not a publisher-supplied provenance manifest or a model's training split.

## Recovered results

| Source dataset | Standardized-image hash match | Shape + correlation candidate | Correlation-only candidate | Total |
|---|---:|---:|---:|---:|
| AMOS | 81 | 117 | 1 | 199 |
| BTCV | 30 | 0 | 0 | 30 |
| FLARE22 | 0 | 50 | 0 | 50 |
| **Total** | **111** | **167** | **1** | **279** |

All 279 rows have unique BDMAP IDs and unique source-dataset/scan pairs. No RAOS matches occur in these recovered exports; that is not proof of no RAOS overlap.

### Which case came from which dataset?

Use [case_mapping.csv](abdomenatlas-provenance/case_mapping.csv). Each row preserves the BDMAP ID, source dataset, original filename, shard tag, matching method, rounded correlation and source export. `evaluation_case_id` translates image names to this repo's evaluation names, for example BTCV `img0009.nii.gz` becomes `btcv_label0009`, and FLARE's `_0000` channel suffix is removed. `evidence_status` keeps hash matches separate from candidates.

Examples from the recovered files:

| BDMAP ID | Source scan | Evidence |
|---|---|---|
| BDMAP_00000089 | AMOS amos_0038.nii.gz | Standardized-image hash match |
| BDMAP_00000220 | BTCV img0009.nii.gz | Standardized-image hash match |
| BDMAP_00000338 | FLARE22_Tr_0014_0000.nii.gz | Shape + correlation candidate |
| BDMAP_00001053 | AMOS amos_0199.nii.gz | Correlation-only candidate, 0.9985; requires confirmation |

## Method and interpretation

The original matcher is preserved in [sources/atlas_match3.py](abdomenatlas-provenance/sources/atlas_match3.py). It clips CT intensities to [-1000,1000], finds the largest 26-connected component within the (-100,100) intensity mask, crops to that component's bounding box, applies its recorded orientation transformation, casts to int16, and computes an MD5 hash of the standardized voxel array. Thus `exact` means an equal hash **after this lossy standardization**, not identical original NIfTI files, matching physical geometry, or a verified patient identifier.

For non-hash matches, the matcher windows to [-200,300], resizes to 32 x 32 x 32, normalizes each descriptor and selects its best correlation against the gold cache. It records equal-shape candidates above 0.97 and other candidates above 0.99. Correlations are saved to four decimals, so `1.0000` is not proof of exact voxel equality. Correlation-based candidates need full-resolution/geometry confirmation before being used as definitive attribution.

The scripts name `BodyMaps/_AbdomenAtlas1.1Mini` and fetch `main` without a pinned revision. The 170 MB `.atlas_std_cache.pkl` remains in the original AdrenalGold folder and is not committed. The copied scripts/notebook are historical evidence, not a portable, newly tested downloader: they contain source-computer paths, incomplete download/completion handling and cleanup operations. Do not run them automatically. The new summary script below only reads the small committed exports.

## Export lineage and incomplete coverage

- `AdrenalGold/atlas_overlap.csv`: 126 rows, shard tags 01-05.
- `AdrenalGold/atlas_shard01_overlap.csv`: 29 rows, already contained in the first file; not counted twice.
- `Downloads/atlas_overlap_0619_shards1to10inclusive.csv`: despite its name, its 142 rows have shard tags 06-10. It is a subset of the later export.
- `Downloads/atlas_overlap_0619_up to shard 13.csv`: 153 rows, with positive-match tags 06-11. Its filename does not prove shards 12-13 completed.
- The preserved Colab notebook proposes processing shards 06-19 but contains no saved execution outputs. The local pipeline log records only part of a shard06 download.

The combined table uses the 126-row first export and 153-row later export. There is no complete processed-shard inventory or negative-match manifest, so missing cases cannot be called non-overlapping or held out. Do not divide these counts by the entire atlas and call the result exhaustive coverage.

## Use alongside model split provenance

See [MODEL_SPLITS.md](MODEL_SPLITS.md) for D997/D998 membership. This mapping identifies possible reuse of public evaluation scans inside AbdomenAtlas 1.1 Mini; it does **not** establish which cases trained the evaluated AtlasNet checkpoint. That checkpoint's recorded dataset provenance is AbdomenAtlas 3.0, and transferring these matches across versions requires an explicit case/image identity check and training-split evidence. This is also distinct from the unrelated AdrenalGold `gold_dl/split_manifest.json` crop-training split and the AA1.1-vs-AA3.0 annotation-comparison experiments.

## Verification

Run `python docs/abdomenatlas-provenance/summarize.py`. It verifies hashes of all recovered source files, checks that earlier exports agree with later ones, rejects duplicate mappings, and rebuilds `case_mapping.csv` and [summary.json](abdomenatlas-provenance/summary.json). [provenance.json](abdomenatlas-provenance/provenance.json) records original paths, hashes and scope. No image matching, cloud workload or new inference was run for this recovery.
