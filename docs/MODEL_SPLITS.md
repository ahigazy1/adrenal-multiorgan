# D997 and D998: recovered split evidence

Audited locally on September 25, 2026. All case IDs below are TotalSegmentator-style `sNNNN` IDs. Model weights are not stored in Git; their pinned Hugging Face links and SHA-256 identities are in the JSON records.

| Model | Training | Validation | Evidence level |
|---|---|---|---|
| D997, fold 0 | **1,082 case IDs recovered** | **57 case IDs verified** | Counts are explicit in the matching final training log. The recovered validation list matches every final-validation case named in that log. The paired training list is corroborated, but not independently enumerated in the model log. |
| D998, fold 0 | Exact IDs **not recovered**. **1,072 inferred**, conditional on the validation summary covering the full fold. | **56 case IDs verified in the saved validation summary** | `dataset.json` records `numTraining=1128`, which includes the training/validation pool. Subtracting 56 gives 1,072; this is not a directly verified training split. |

## Files to use

- [D997.json](model-splits/D997.json): recovered `train` and `val` lists, confidence labels, count evidence and checkpoint link. This is a reconstruction, **not** the original `splits_final.json`.
- [D998.json](model-splits/D998.json): validation IDs and checkpoint link. `train` is deliberately `null`, not an empty training set and not a copy of D997's training list.
- [provenance.json](model-splits/provenance.json): original local locations, source hashes, search scope and validation-set comparison.
- [sources/](model-splits/sources/): original D997 case-list text files, matching final training log, dataset/debug records, and D998's full saved validation summary. The raw summary retains its original JSON `NaN` values; the normalized model records contain no nonstandard values.

Run `python docs/model-splits/check.py` to verify source hashes, counts, disjointness and validation membership directly against the included evidence. No external files or packages are required.

## What establishes the model identities

D997 was found in `Downloads/Dataset997_ChestAbd_ResEncUNetL_8000epochs_NoMirroring.zip`. Use the `nnUNetTrainer_8000epochs_NoMirroring__nnUNetResEncUNetLPlans__3d_fullres` directory, not the preliminary `NoMirroringFinite` directories. Its final checkpoint SHA-256 was recomputed as `0840b5b15364daa194033cef1466b522316c440cc190c80fdf7091dca02c17f7`, the same checkpoint used by the corrected-reader `labmate997-fixed` evaluation.

The final D997 log, `training_log_2026_9_14_14_39_39.txt`, explicitly records one split, fold 0, 1,082 training cases and 57 validation cases. It later names all 57 validation scans. The separate `split0_validation_samples.txt` matches those IDs exactly. Its paired 1,082-case training list has no duplicates or validation overlap. Both text files are byte-identical in two downloaded copies of the model-development project. This supports the recovered training list, but cannot substitute for a missing original split file when absolute training-membership certainty is required.

D998 was found in `Downloads/models_archive/Dataset998_TotalSeg66classes/nnUNetTrainer_2000epochs_NoMirroring__nnUNetResEncLPlans_16GB__3d_fullres_bs8`. Additional copies exist under `Desktop/Training_Run/models_archive`, `Documents/CoLab` and `Desktop/adrenalSegmentator`. The Downloads checkpoint SHA-256 was recomputed as `a54d14deeb70f7b8f9e34739f50efb740a3193829e7c19325342681fc8fe396c`, matching the evaluated D998 checkpoint. Its `fold_0/validation/summary.json` names 56 reference/prediction pairs. Internal dataset paths use `Dataset999_TotalSeg66classes`, despite the published Dataset998 folder name.

An older nearby March 2025 log reports a 1,072/56 split, but names a **different** `Dataset999_CustomTotalSeg` model with a ResEncM plan. Its relevant lines are included as supporting context and must not be described as D998's own training log.

## Relationship and limits

All 56 D998 validation IDs occur in D997's 57-case validation list. D997 additionally includes `s0352`. This does **not** establish that the two training lists match, or identify the ten training-case differences suggested by the counts.

No original D997/D998 `splits_final.json` or model-specific held-out test manifest was recovered. The standalone `Downloads/splits_final.json` and `splits_final adrenal.json` instead contain BDMAP case IDs and are unrelated. Neither omission from a recovered list nor the generic model-development `test_samples.txt` proves a case was unseen by these checkpoints.

For overlap audits, use D997's recovered training membership with its stated confidence and use the two verified validation memberships. Treat other D998 cases as **unknown**, not training or test by default. Do not automatically replace Dataset902/903 splits with these historical lists. The existing AtlasNet fine-tune test set is not automatically held out from D997 or D998 pretraining.
