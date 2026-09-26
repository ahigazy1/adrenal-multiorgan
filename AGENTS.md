# AGENTS.md: picking up this project

Read this first. It is the operational state as of **2026-09-26 02:25 UTC**, written so another agent (or person) can continue without the chat history. When anything here disagrees with live state, live state wins: re-check before acting.

## Ground rules from the user

- **One training run only.** Nothing may be launched until the fine-tune design is agreed with the user. Smoke tests and checks are fine; the real run needs explicit approval.
- **No paid-resource launches or deletions without asking.** Never delete the `adrenal-train` VM or its disk.
- Ponytail style: reuse existing code, stdlib first, minimum code, Polars over pandas, compact factual reports, simple operator commands. Every claim should come from a log, file or command output, not memory.
- **Do not edit these files:** `cohort_scan.py`, `build_dataset.py`, `preprocess.py`, `prepare.sh`, `atlasnet_plans.json`, `pixi.lock`, `vendor/`, `trainers/adrenal_multiorgan.py`. Their hashes fingerprint the finished Dataset902 run (`train.py` `run_record`) and its preprocessing cache (`hf_cache.recipe_id`). Editing them breaks resume and prediction for that run. Subclass or wrap instead, as `finetune.py` and `trainers/d997_finetune.py` do.
- Commit identity: this computer has no git identity. Use a per-command identity; do not change git config:
  `git -c user.name=ahigazy1 -c user.email=168588872+ahigazy1@users.noreply.github.com commit ...`
  For a rebase, export `GIT_AUTHOR_NAME/EMAIL` and `GIT_COMMITTER_NAME/EMAIL` instead. End commit messages with the Co-Authored-By line used in `git log`.
- The user sometimes pushes from another session. `git fetch` before every commit, and rebase instead of force-pushing.

## Models and what is settled

| Name | What | Where |
|---|---|---|
| D997 | nnU-Net ResEncL, 66 classes, 1.5 mm, 8,000 epochs, trained on TotalSegmentator only | HF `ahigazy1/adrenal-training-models` `labmate/` @ `3a52ec5a`, checkpoint sha256 `0840b5b1...`. Needs the reorienting reader (`NibabelIOWithReorient`); its plans say `NibabelIO`, which is the known bug. |
| D998 | nnU-Net ResEncL, same 66 classes and IDs, 2,000 epochs | HF `ahigazy1/AdrenalSeg-Sources` `Dataset998_TotalSeg66classes/...` @ `873ffe7a` |
| AtlasNet | nnU-Net, trained on all of AbdomenAtlas 3.0 | HF `AbdomenAtlas/AtlasNet` @ `fd03b410` |
| AtlasNet fine-tune (`ours-epoch1957`) | our Dataset902 run, 9 classes, 1 mm, focal Tversky | HF `ahigazy1/adrenal-multiorgan-model` |

**Splits for every model:** see [docs/MODEL_SPLITS.md](docs/MODEL_SPLITS.md). `python docs/model-splits/check.py` recomputes and verifies them.
- **D997:** 1,082 train / 57 validation, recovered lists.
- **D998:** 1,072 train, reconstructed uniquely from its fingerprint (D997's training list minus 10) / 56 validation.
- **AtlasNet:** AbdomenAtlas contains 204 of 300 AMOS, all 30 BTCV and all 50 FLARE22 scans, including **53 of the 75-scan comparison set**. Two independent methods agree; see also [docs/ABDOMENATLAS_PROVENANCE.md](docs/ABDOMENATLAS_PROVENANCE.md).
- **D997 and D998 never saw AMOS/BTCV/FLARE**, so the 75 scans are truly held out for them. They did see about 93% of TotalSegmentator.

**Pseudo-labels (done):** TotalSegmentator 2.18, D997's 66 classes, 380 labelled scans (AMOS 300, BTCV 30, FLARE22 50).
- **Where:** HF `ahigazy1/adrenal-pseudolabels` (public for now; the user will make it private later), prefix `pseudolabels/totalseg-2.18.0/bfece4a56f11d1f1`, commit `3421a9d`. Uses `roi_subset` (runs only the organ, vertebra and cardiac models).
- **Accuracy vs full TotalSegmentator:** equivalent on AMOS ground truth (Dice 0.7995 vs 0.8004).
- **Score vs ground truth:** `evaluation/` in the same repo. Adrenal Dice is about 0.71, so real labels must override pseudo-labels.

**Comparisons (done):** on HF `adrenal-multiorgan-model`:
- `comparison/`: the 279-scan test set
- `comparison-external/`: all 330 AMOS+BTCV
- `comparison-flare/`: all 50 FLARE22, D997 and D998
- `comparison-raos/`: partial; not a test set, since the user says RAOS ground truth over-segments

## Fine-tune: decided vs open

**Decided by the user:**
- D997 at 1.5 mm
- CE + focal Tversky loss (0.4 FP / 0.6 FN, exponent 0.75)
- stock nnU-Net augmentation **without mirroring**
- include TotalSegmentator scans
- RAOS **for training only**
- one run

**Built but paused. Do not run until the design is agreed:**
- **`finetune.py`:** builds Dataset903 and preprocesses it with D997's plans.
  - **Labels:** real labels are authoritative; pseudo-labels fill the other classes; adrenals are written last; kidney cysts merge into kidneys.
  - **Revised exclusion rule:** glands cut off by the scan edge are kept. Drop a scan only for a fragmented gland, a fully visible gland ≤ 1 mL, or a failed left/right check.
  - **Split:** Dataset902's test and validation sets, unchanged.
  - **Provenance:** writes `provenance.csv` and `provenance.json`.
- **`trainers/d997_finetune.py`, `ADRENAL_RUN=d997 python train.py`, and `gcp/finetune_startup.sh`.**
- **Partial build to delete first:** a build was started and then killed. `/opt/adrenal-multiorgan/data/nnUNet_raw/Dataset903_D997Finetune` holds 189 partial label files and no finished marker. Delete that folder before any rebuild.

**Must fix before building** (from the code review; details in the handoff folder's DESTINATION_STATE.md):
1. Clear `Dataset903` at build start, and put the git SHA in its finished marker.
2. Score test scans only on each source's real-label organs.
3. Remove the shutdown placeholder (see the VM section) and restore the training startup script before the run.
4. Set the data-loader environment variables in `finetune_startup.sh`, as `gcp/startup.sh` does.
5. Make `train.py` reject any `ADRENAL_RUN` other than `atlasnet` or `d997`.
6. Delete any smoke-test model folder before the real run: `EPOCHS` is part of the run fingerprint.
7. Fix provenance labels for scans that failed the initial scan (they're recorded as excluded for the wrong reason).
8. Replace `EPOCHS = 1000` in `d997_finetune.py` with a value based on the smoke test's seconds per epoch.

**Open decisions (ask the user):**
- (a) Test only on the 75 AMOS/BTCV/FLARE scans for D997-based models, with TotalSegmentator test scans reported separately as "seen in pretraining"?
- (b) Validate and select checkpoints on AMOS/BTCV/FLARE plus the TotalSegmentator scans neither D997 nor D998 used (89 scans)?
- (c) Sample about 50/50 between AMOS/BTCV/FLARE and TotalSegmentator?
- (d) Which pseudo-label teacher, TotalSegmentator or VISTA3D, per class? Wait for the teacher comparison.
- (e) RAOS labels: RAOS ground truth over-segments per the user. Use its labels (all, or all except adrenals), or only its images with pseudo-labels?

## In progress when this was written

- **VISTA3D pseudo-labels** for the 380 scans (`pseudolabels/vista3d/run.sh`): outputs in `/opt/vista/out/<shard>/<case>/<case>_trans.nii.gz`, raw VISTA3D label IDs. It was at 291 of 380. Logs: `/var/log/vista3d.log` and `/var/log/vista3d-{0..3}.log`. The PyTorch 2.9+ indexing warning is harmless: outputs were checked against FLARE labels.
- **Teacher comparison** (`pseudolabels/vista3d/compare_teachers.py`): a local background loop starts it automatically when VISTA3D finishes. It writes `/opt/pseudo/evaluation/teachers_vs_ground_truth.csv` and prints a per-organ table. **If the loop died, run it by hand:**
  copy `pseudolabels/vista3d/compare_teachers.py` to the VM as `/tmp/compare_teachers.py` (base64 pattern below), then `sudo /opt/pseudo-env/bin/python /tmp/compare_teachers.py`. It reads `/opt/vista/bundle/docs/labels.json`.
  VISTA3D was trained on AMOS and TotalSegmentator, so compare the teachers mainly on FLARE22.
- **RAOS:** downloaded to `/opt/adrenal-multiorgan/data/raos-source/RAOS-Real` (826 files: 413 scans plus labels), AdrenalSeg-Sources @ `873ffe7a`. There are no pseudo-labels for RAOS yet: it needs TotalSegmentator (add a `raos` source to `pseudolabels/pseudolabel.py`; its old RAOS layout code is in commit `29c4003`) and VISTA3D (link the RAOS images into a new input folder).

## Google Cloud and the VM

- **gcloud account** `zakiyaferdousi@gmail.com`, **project** `adrenal-seg`. HF token: Secret Manager secret `HF_TOKEN`, readable by the account and by the VM's service account. **Never print it or save it to disk.**
- **VM `adrenal-train`**, us-central1-f: g4-standard-48 Spot (RTX PRO 6000, 96 GB), 500 GB disk (about 179 GB free). **It is RUNNING and billing.** Only us-central1-f works (the disk is zonal); if Spot capacity is short, retrying later works.
- **Why nothing else can be created:** the project's Hyperdisk quota is 500 GB, all used by this disk, and increase requests were auto-denied. No second VM is possible.
- **Current VM state, which matters:**
  - **Startup script:** the startup-script metadata is the **pseudo-label** script, not the training one. Before training, restore it: [../../ops/pseudo-on-adrenal-train/README.md](../../ops/pseudo-on-adrenal-train/README.md) in the handoff folder, `original-training-startup.sh`. Or set `gcp/finetune_startup.sh` with metadata `finetune-revision=<full 40-character sha>`.
  - **Shutdown placeholder:** `/root/.local/bin/shutdown` stops startup scripts from powering off the VM, because the user asked to keep it running. It only takes effect where `/root/.local/bin` is first in `PATH`. **Remove it** (`sudo rm /root/.local/bin/shutdown`) so runs power off when done.
  - **Metadata keys:** `pseudo-revision`, `pseudo-workers` (safe to remove after restoring).
- **What's on the disk:**

| Path | Contents |
|---|---|
| `/opt/adrenal-multiorgan` | Frozen training checkout plus `data/`: TotalSegmentator zip, AMOS/BTCV/FLARE sources, Dataset902 raw and preprocessed, `raos-source`, `d997`. Leave the checkout alone. |
| `/opt/pseudo`, `/opt/pseudo-run`, `/opt/pseudo-env` | Pseudo-labels (all attempts), their code, and their Python environment |
| `/opt/flare-run` | FLARE comparison checkout; its pixi environment has `psutil` added via uv |
| `/opt/finetune-run` | Fine-tune checkout at `8920e4a`, pixi environment |
| `/opt/vista` | VISTA3D environment (uv; torch cu128, monai 1.4.0, pytorch-ignite), bundle, `in/` symlinks, `out/` |
| `/opt/atlas-check` | 18 extracted AbdomenAtlas scans, safe to delete |

### Running commands on the VM from this Windows computer (Git Bash)

- **The `gcloud` Bash wrapper picks up the Windows Store Python placeholder.** Always set:
  `export CLOUDSDK_PYTHON="/c/Users/zakiy/AppData/Local/Google/Cloud SDK/google-cloud-sdk/platform/bundledpython/python.exe"`
  (PowerShell's `gcloud` works without this.)
- **Command pattern:**
  `timeout 200 gcloud compute ssh adrenal-train --zone=us-central1-f --quiet --command="..." < /dev/null 2>/dev/null`
  `gcloud compute ssh` uses PuTTY's `plink` here.
  - **Host-key prompt:** it prints a host-key prompt that **eats stdin**, so never pipe a script into ssh. Instead, base64-encode it locally and decode it on the VM:
    `B=$(base64 -w0 script.py); ... --command="echo $B | base64 -d > /tmp/s.py && sudo /opt/pseudo-env/bin/python /tmp/s.py"`
  - **Output noise:** filter the host-key chatter with `grep -vE "host key|Plink|trust|carry on|fingerprint|ssh-ed25519|guarantee|think it is|Store key|cancels|port 22|enter \"n\"|^connection"`.
- **Long jobs:** write a script to `/opt/<name>.sh` that logs with `exec > >(tee -a /var/log/<name>.log) 2>&1`, then start it detached:
  `(sudo setsid nohup /opt/<name>.sh >/dev/null 2>&1 < /dev/null &)`
- **`pkill -f` pitfall:** `pkill -f <pattern>` also matches the ssh command's own shell (exit code 128). Use a bracket pattern such as `pkill -f '[f]inetune.py build'`. Killing a parent script does not kill its multiprocessing workers; kill those too.
- **HF inside VM jobs:** get the token from the metadata server, as `gcp/startup.sh` does. For big downloads use `hf_hub_download` / `snapshot_download` with `HF_XET_HIGH_PERFORMANCE=1` (fast). `HfFileSystem` streaming is slow.
- **Monitoring and logs:** Cloud Logging keeps the serial console (startup scripts) after the VM stops:
  `gcloud logging read 'resource.type="gce_instance" AND textPayload:"..."' --project=adrenal-seg`
  **Pseudo-label monitor** (read-only, Windows PowerShell, from the handoff folder):
  `.\.venv-check\Scripts\python.exe .\projects\adrenal-multiorgan\pseudolabels\monitor.py --instance adrenal-train --zone us-central1-f`
- **Local Python:** use the handoff folder's `.venv-check\Scripts\python.exe`; plain `python` is the Windows Store placeholder.

## Hugging Face storage (user is near the limit)

- **Private:** 818 GB = 587 GB current files + 231 GB history, mostly overwritten checkpoints: `adrenal-atlas-checkpoints` 116 GB, `aa11-*-08mm-checkpoints` 37 GB, `adrenal-multiorgan-model` 25 GB.
- **Public:** 2.25 TB, mostly current files (`adrenal-atlas-preprocessed` 1.33 TB, `adrenal-atlas-aligned` 493 GB).
- **`testing`:** 12 GB, all history.
- **Pending decision:** squashing history (`HfApi.super_squash_history`) or deleting anything is irreversible and awaits the user. The fine-tune trainer uploads every 250 epochs to limit growth.
- **Getting storage figures:** `usedStorage` works through the raw API (`/api/models/<id>?expand[]=usedStorage`), not through `huggingface_hub`'s listing.
