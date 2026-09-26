# Operating guide

Updated September 26, 2026. Recheck live logs before reporting progress. Historical chronology is in the handoff's `DESTINATION_STATE.md`; this file records current rules and locations.

## Rules

- One new training run only, after the user agrees the design. No training or paid-resource launches without approval. Never delete `adrenal-train` or its disk.
- Preserve local changes. Reuse existing code, prefer stdlib and Polars, and keep implementations small.
- Do not edit `cohort_scan.py`, `build_dataset.py`, `preprocess.py`, `prepare.sh`, `atlasnet_plans.json`, `pixi.lock`, `vendor/`, or `trainers/adrenal_multiorgan.py`: their hashes fingerprint the finished Dataset902 run/cache. Wrap or subclass instead.
- Result overviews: left/right adrenals, aorta, pancreas and liver; Dice plus signed/absolute volume errors.
- Before every commit, fetch; another session may push. Rebase rather than force-push. Use per-command identity: `git -c user.name=ahigazy1 -c user.email=168588872+ahigazy1@users.noreply.github.com commit ...`. End messages with the Co-Authored-By convention in `git log`. Do not change git config.
- Never print or persist HF tokens. No history squashing or remote deletion without approval.

## Models and evidence

| Model | Identity |
|---|---|
| D997 | 66 classes, 1.5 mm, 8,000 epochs; HF `ahigazy1/adrenal-training-models/labmate` @ `3a52ec5a`; checkpoint `0840b5b1...`. Inference requires `NibabelIOWithReorient`, correcting its published reader. |
| D998 | 66 classes, 1.5 mm, 2,000 epochs configured; HF dataset `ahigazy1/AdrenalSeg-Sources` @ `873ffe7a`, `Dataset998_TotalSeg66classes/`. |
| AtlasNet | AbdomenAtlas 3.0; HF `AbdomenAtlas/AtlasNet` @ `fd03b410`. |
| AtlasNet fine-tune | Dataset902, nine classes, 1 mm, focal Tversky, 2,000 epochs finished. Evaluate `ours-epoch1957`; HF `ahigazy1/adrenal-multiorgan-model`. |

- [MODEL_SPLITS.md](docs/MODEL_SPLITS.md): D997 1,082 recovered train / 57 validation; D998 1,072 reconstructed train / 56 validation. Check with `python docs/model-splits/check.py`.
- [ABDOMENATLAS_PROVENANCE.md](docs/ABDOMENATLAS_PROVENANCE.md): AtlasNet pretraining overlaps 204 AMOS, all 30 BTCV and all 50 FLARE scans, including 53 of matched75. D997/D998 did not train on those sources; they did train on most TotalSegmentator scans.
- Finished HF comparisons: `comparison/` (279), `comparison-external/` (AMOS300/BTCV30), `comparison-flare/` (FLARE50, D997/D998). RAOS is training-only, not a held-out benchmark.

## Active AtlasNet inference and next step

- All 1,228 TotalSegmentator v2.0.1 scans, four model replicas; each uses three preprocessors, two exporters and two CPU threads. This is inference of the finished model.
- Runner `gcp/atlasnet_totalseg.py`; VM `/tmp/atlasnet-totalseg.py`, Python `/opt/flare-run/.pixi/envs/default/bin/python`.
- Checkpoint SHA-256 `f25e3161e1b1e8ea6beb289db7a0feb083d49b4ed5abde7bb7c63459861f598e` matches evaluated epoch1957 (saved counter 1958).
- Batching check passed: 3 / 41,869,872 voxel labels changed versus stock nnU-Net. Four-case native-grid/upload smoke passed. At 04:49 UTC, 520/1,228 uploaded, GPU 100%; completion still pending at that snapshot.
- Run root `/opt/atlasnet-totalseg`; main log `/tmp/atlasnet-totalseg.log`; per-worker logs at the run root. `complete.json` is written after all masks and remote checks succeed.
- HF prefix: `ahigazy1/adrenal-multiorgan-model/comparison-totalseg/ours-epoch1957/`.
- Next: [raw-reference scoring by Dataset902 split](docs/TOTALSEG_EVALUATION.md), using `evaluate_totalseg.py`. `gcp/atlasnet_totalseg_postprocess.py` is queued to wait for completion, score and checksum-verify uploaded CSVs. Staged in `/opt/atlasnet-totalseg/scoring/`; synthetic and two-case real-data checks passed. Log `/tmp/atlasnet-totalseg-postprocess.log`.

## Teacher experiments

- **TotalSegmentator 2.18:** 380 AMOS/BTCV/FLARE masks, D997's 66 IDs. HF dataset `ahigazy1/adrenal-pseudolabels`, prefix `pseudolabels/totalseg-2.18.0/bfece4a56f11d1f1`, commit `3421a9d`; metrics under `evaluation/`.
- **VISTA3D:** original 380 plus all 413 RAOS masks complete. HF prefix `pseudolabels/vista3d-c6dbe159/`. RAOS: 317 cancer, 22 partial-surgery, 74 missing-organ scans; all image/label grids matched. Left/right adrenal mean Dice .620/.595; median signed volume errors −35.1%/−31.7% against RAOS labels.
- **CTMR:** three-case RAOS trial complete and uploaded under `pseudolabels/nv-segment-ctmr-4fb8b4a6/raos-trial/`. Nine-case AMOS/BTCV/FLARE trial also complete, at `/opt/ctmr/abdominal-trial`; its upload is pending. Upstream `cb921f5c...`, weights `4fb8b4a6...`.
- The combined 810-file teacher upload was checksum-verified at HF commit `48164b34b240caa2141aafa9830db9368a957db2`; manifest `manifests/teacher-results-20260926.json`. Handoff uploader: `ops/upload-teacher-results.py`.
- **NV-Segment-CT points:** six-gland/three-scan automatic-versus-one-point trial complete at `/opt/nvct/`; weights `afb51518...`. Volume accuracy worsened in five of six glands. AMOS0005 left prediction 18.01 mL versus 0.67 mL reference. Geometry audit confirmed the point was inside the reference and correctly transformed. Class-plus-point mode merges automatic/interactive predictions; it did not refine an imported CTMR mask. Inspection evidence is in handoff `verification/takeover-evidence/nvct-point-inspection.md`. Upload pending. Earlier `ops/vista-point-trial.py` was never launched.
- CT supports point prompts; CTMR officially does not. Reference-assisted results must be distinguished from automatic inference. Both NVIDIA teachers have source-dataset exposure; these trials do not establish unseen-data performance.

## D997 fine-tune: paused

Decided: D997 at 1.5 mm; CE + focal Tversky (FP .4 / FN .6, exponent .75); stock augmentation without mirroring; include TotalSegmentator; RAOS as a normal labelled, training-only dataset; one run.

Open: sampling ratios, final validation/test membership, teacher choice for missing organs, epoch budget and checkpoint selection. Volume-aware full-volume validation has been discussed, not implemented or approved as the selection rule.

`finetune.py`, `trainers/d997_finetune.py` and `gcp/finetune_startup.sh` are a paused proposal. The builder currently handles TS/AMOS/BTCV/FLARE, not RAOS. It uses Dataset902 validation/test membership and keeps truncated glands under its revised exclusion rule. This is not the final agreed design.

Before a build/run:
1. Resolve existing-output handling and record the code revision in the build marker.
2. Score only real-label organs for each source.
3. Correct provenance for scans rejected during the initial audit.
4. Set loader environment variables, agree epoch budget, and remove any smoke-model output before training.
5. Restore training startup metadata and remove the shutdown shim described below.

The abandoned Dataset903 build was removed during approved cleanup on September 26. No new training run has started.

## VM and commands

- Account `zakiyaferdousi@gmail.com`, project `adrenal-seg`; VM `adrenal-train`, zone `us-central1-f`, g4-standard-48 Spot: RTX PRO 6000 96 GB, 48 vCPUs, 176 GiB RAM, 500 GB retained disk. Last checked RUNNING and billing. Reuse this VM; disk quota is exhausted.
- PowerShell SSH: `gcloud compute ssh adrenal-train --zone=us-central1-f --project=adrenal-seg --quiet --command="..."`. Upload scripts with `gcloud compute scp`; avoid nested Python quoting or piping scripts through PuTTY stdin.
- Git Bash needs `CLOUDSDK_PYTHON="/c/Users/zakiy/AppData/Local/Google/Cloud SDK/google-cloud-sdk/platform/bundledpython/python.exe"`. PowerShell does not.
- Detached jobs: `(sudo setsid nohup <command> >/tmp/<job>.log 2>&1 < /dev/null &)`. An SSH monitor disconnect does not stop these jobs. Killing a parent does not kill its worker processes.
- Startup metadata still points to the pseudo-label script. Original training startup is in handoff `ops/pseudo-on-adrenal-train/original-training-startup.sh`; D997 uses `gcp/finetune_startup.sh` with a pinned revision.
- `/root/.local/bin/shutdown` is a placeholder installed at the user's request to keep the VM running. Restore real shutdown behavior before an approved training launch. Do not change the current inference lifecycle silently.
- HF authentication: Secret Manager `HF_TOKEN`, readable by the VM service account. Use metadata-server credentials as in `gcp/startup.sh`; hold the token in memory. `HF_XET_HIGH_PERFORMANCE=1` speeds large downloads.
- Local Python: handoff `.venv-check/Scripts/python.exe`; plain `python` is a Windows Store alias.

| VM path | Contents |
|---|---|
| `/opt/adrenal-multiorgan` | Frozen training checkout and original data, Dataset902, checkpoints, RAOS and D997. Preserve. |
| `/opt/flare-run` | Comparison checkout and pixi runtime used for AtlasNet inference. |
| `/opt/finetune-run` | Paused fine-tune checkout. |
| `/opt/pseudo`, `/opt/pseudo-run`, `/opt/pseudo-env` | TotalSegmentator outputs, code and runtime. |
| `/opt/vista` | VISTA3D environment/bundle, original `out/`, RAOS `raos/`. |
| `/opt/ctmr`, `/opt/nvct` | Isolated NVIDIA trial runtimes and evidence. |

Approved cleanup removed the abandoned Dataset903 build and disposable `/opt/atlas-check` scans; replaced 380 checksum-identical duplicate CTs with symlinks to retained originals, reclaiming ~24.2 GiB. Preserve their targets. Log `/var/log/adrenal-cleanup-20260926.json`; handoff copy `verification/takeover-evidence/vm-cleanup-log.txt`.

HF storage was near its limit at the last audit, mostly checkpoint history. No squashing/deletion is approved. Use raw API `usedStorage` for new audits; old usage snapshots are not current quotas.

## VM disk archive (2026-09-26)

Everything that existed only on the `adrenal-train` disk was uploaded to HF `ahigazy1/adrenal-multiorgan-model` under `vm-archive-2026-09-26/` (384 files, 79 MB, all verified by recursive listing). It contains:
- the Dataset902 final-checkpoint test predictions
- all run logs
- the full-TotalSegmentator AMOS outputs
- the CTMR and NV-CT teacher trials
- the ad hoc job scripts

Everything else on that disk was already on HF (the Dataset902 cache, the model folder, VISTA3D for 380 + 413 RAOS scans, AtlasNet fine-tune on TotalSegmentator under `comparison-totalseg/`), public, or reproducible.

**HF listing gotcha:** on repos this large, `repo_info(...).siblings` returns a truncated list. Use `list_repo_tree(recursive=True)` to verify files.
