# AtlasNet fine-tune: all TotalSegmentator scans

The current inference run uses the completed Dataset902 epoch-1957 checkpoint on all 1,228 TotalSegmentator v2.0.1 CTs, with four model replicas. No mirroring. Outputs retain Dataset902's nine foreground IDs.

## Score completed predictions

```bash
python evaluate_totalseg.py --run /opt/atlasnet-totalseg --data /opt/adrenal-multiorgan/data --workers 8
```

Requires inference `complete.json`, `cases.json`, masks under `worker-*/native_labels/`, the source ZIP, Dataset902 `build_dataset.csv` and fold-0 `splits_final.json`. The scorer verifies prediction hashes, label IDs, grids and split consistency. Eight CPU processes read one organ at a time; no extracted reference dataset is needed.

Outputs under `evaluation/`:
- `per_case.csv`: nine organs per scan, Dice, reference/predicted volume, signed and absolute volume errors in mL and percent.
- `by_split.csv`: training, validation, test and excluded-from-Dataset902 summaries.
- `focus_by_split.csv`: left/right adrenals, aorta, pancreas and liver.
- `all_scans_descriptive.csv`: pooled descriptive results, not a held-out estimate.
- `complete.json`: input/code/output hashes and metric definitions, written last.

References are raw per-organ masks: kidney cysts merge into kidneys, adrenal speckles remain, overlapping source masks are scored independently. These results are not directly interchangeable with scores against Dataset902's cleaned labels. Percent error is null when reference volume is zero; Dice is null only when both masks are empty. No surface-distance metrics are computed by this scorer.

Dataset902 membership describes fine-tuning exposure, not proven absence from AtlasNet pretraining.

## Current VM workflow

`gcp/atlasnet_totalseg.py` is the inference runner, deployed on this VM as `/tmp/atlasnet-totalseg.py`. `gcp/atlasnet_totalseg_postprocess.py` waits for inference completion, runs the scorer, uploads CSVs to `ahigazy1/adrenal-multiorgan-model/comparison-totalseg/ours-epoch1957/evaluation`, downloads that exact upload revision to verify hashes, and publishes the completion manifest last. It stops if inference exits without its completion marker.

The postprocessor is queued on the VM. Its staged code is under `/opt/atlasnet-totalseg/scoring/`, with `PYTHONPATH=/opt/flare-run` for the unchanged dataset modules. Synthetic checks and a two-case raw-reference smoke passed before queueing.

Inference log: `/tmp/atlasnet-totalseg.log`. Postprocessing log: `/tmp/atlasnet-totalseg-postprocess.log`.
