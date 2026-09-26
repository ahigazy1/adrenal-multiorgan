# Pseudo-label generation

Generate raw predictions for the scans that have ground-truth labels: **300 AMOS CT, 30 BTCV and 50 FLARE22 (380 total)**. Unlabelled AMOS test scans and RAOS are excluded. This stage does not merge masks, select new splits, or train a model. The intended later model is D997 or D998; fine-tuning design remains a separate step.

## Label identity and geometry

The saved D997 and D998 dataset.json files were compared on September 25, 2026: both use the same 66 foreground names and numeric IDs, exactly matching `D998` in pseudolabel.py. TotalSegmentator's standard total task has 117 classes. Map by **name**, never copy numeric IDs directly: for example, TotalSegmentator aorta=52 becomes D997/D998 aorta=50. Kidney cysts are merged into their corresponding kidney; unrelated classes become background. `recipe.label_ids`, `recipe.source_label_ids` and `recipe.merged` document this conversion.

Each output is a uint8 NIfTI on its source CT grid, retaining the input qform/sform. TotalSegmentator keeps its own canonical orientation conversion, crop reversal and native-grid restoration. A mismatched output shape or affine and invalid IDs fail the job.

## CPU and GPU scheduling

Default: **24 case workers, 2 CPU threads per worker**, each with its own nnU-Net slot (the semaphore equals the worker count). Per-scan nnU-Net preprocessing and export are mostly single-threaded CPU and the GPU part takes about a second, so capping concurrent invocations at 1 or 4 left the machine mostly idle. Uploads happen every 48 scans (two worker waves). TotalSegmentator is called with `roi_subset` = the 66 classes plus kidney cysts, so it runs only its organ, vertebra and cardiac models (tasks 291-293) after its 6 mm crop model (298), skipping the ribs and muscles models whose classes all become background. Ribs/muscles therefore no longer overwrite contested voxels, and inputs are cropped to the ROI plus 20 voxels. The gate covers model loading, inference and nnU-Net export. CPU preparation and final mask processing can overlap across cases; workers waiting for the GPU do not consume four busy cores. Each process can establish a CUDA context, but model inference is serialized and the CUDA allocator cache is released after each invocation. Actual RAM/VRAM use and throughput still need a real-machine check.

`PSEUDO_WORKERS` accepts 1–32. OMP, MKL, OpenBLAS, SimpleITK and nnU-Net export thread defaults are set to 2. Nested nnU-Net preprocessing/export pools are restricted to one process each. This bounds individual pools; 12 x 4 is a CPU scheduling budget, not a guarantee of constant 100% utilization or an exact global OS thread limit.

`runtime.py` replaces TotalSegmentator's outer array resampling with SimpleITK: linear for CT, nearest-neighbour for categorical masks. It retains scipy.zoom's endpoint-aligned coordinate convention and the teacher's surrounding affine/orientation handling. The optional outer cuCIM path uses the same CPU hook. This does **not** replace the trained model's internal nnU-Net plans/resampler or its reader. The pinned TotalSegmentator 2.18.0 API was inspected; other versions require review.

## Deployment and resume

The safeguards and runtime adapter are local changes. Commit and publish the complete reviewed snapshot before launch, including runtime.py, requirements.txt and check_pseudolabel.py. The launcher refuses dirty deployment files or a HEAD not published in origin/main history. It writes the commit SHA to pseudo-revision metadata; startup checks out that exact commit, and resume refuses a different revision. No cloud launch is authorized by this document.

After deployment and explicit launch authorization, the Cloud Shell command is:

```bash
git -C adrenal-multiorgan pull --ff-only && PSEUDO_WORKERS=12 bash adrenal-multiorgan/pseudolabels/gcp/create_vm.sh
```

On subsequent resumes, use the same published checkout/revision rather than pulling newer code. The first scan must pass real inference, native-grid checks, upload and remote checksum verification before batches of 48. Failure exits nonzero and startup shuts down the instance. The retained disk remains allocated. Progress: `NAME=adrenal-pseudo bash adrenal-multiorgan/gcp/progress.sh`; log: `/var/log/pseudo.log`.

Outputs go to the dedicated public HF dataset `ahigazy1/adrenal-pseudolabels`, prefix `pseudolabels/totalseg-2.18.0/<recipe>/`. The source subset, label maps, resampling implementation, versions, code and weights identify the recipe. These changes get a new prefix and cannot resume the old 793-scan/default-resampler recipe. Only hash-verified remote manifest entries are skipped; complete requires all 380 outputs.

## Checks

Run `python pseudolabels/check_pseudolabel.py`. It checks nested multiprocessing, source pairing, output grids, ID conversion, upload failures/corrupt resume, SimpleITK interpolation against scipy, and GPU-gate cleanup. These are synthetic offline checks, not evidence of CUDA operation, anatomical accuracy or 12-worker speedup.

## Local progress monitor

`monitor.py` uses only Python's standard library. It is read-only: it never creates, restarts or stops a VM, and never writes to HF. No extra HF login or saved local token is needed on this computer. It uses an existing `HF_TOKEN` environment variable if supplied; otherwise it reads the project's `HF_TOKEN` secret through your signed-in gcloud account and keeps the token only in memory.

From the extracted handoff root in PowerShell:

```powershell
& .\.venv-check\Scripts\python.exe .\projects\adrenal-multiorgan\pseudolabels\monitor.py
```

With a normal Python install, from the active repository:

```text
python pseudolabels/monitor.py
```

Add `--once` for one check, `--json` for machine-readable output, or `--interval 30` for 30-second polling (default 60). Ctrl+C stops the monitor without affecting inference. `--no-cloud` skips VM/log requests. No uv installation is necessary.

The monitor displays checksum-matched remote uploads out of 380 and per-source counts, VM status and recent serial/Cloud Logging messages. Case-start/case-done logs show work between commits; a case-done log is not an uploaded result. Upload progress advances after the first case, then after batches of 48. A stopped VM is never treated as successful completion without the full verified manifest. No current manifest displays WAITING_FOR_MANIFEST, not completion.

The monitor matches the local runner/runtime/requirements hashes, accepting Git's LF/CRLF conversion, so it does not mix old recipes. Once selected, the run prefix is atomically bookmarked in `.pseudolabel-monitor.json` (gitignored, no credentials). Rerun the same command to resume monitoring. Every check rereads remote truth rather than trusting cached counts. Use `--state other-file.json` for a separate run, or `--prefix pseudolabels/totalseg-2.18.0/RECIPE_HASH` to follow an older recipe explicitly after changing local code.

Inference resumes from hash-verified remote manifest entries. Source downloads and the boot disk are retained. An interrupted, uncommitted batch may be recomputed (up to 25 cases); that is intentional rather than trusting unchecked local files. Pinned startup code prevents automatic upgrades from changing recipe identity on reboot. Restart monitoring independently of restarting inference; the monitor itself never restarts the VM.

Monitor tests: `python pseudolabels/check_monitor.py`. They cover partial/completed manifests, corrupt payloads, false completion, saved-run restart and refreshed remote counts after reconnecting.

## Public output storage

The user requested public pseudo-label uploads while private storage is constrained. The dedicated dataset is https://huggingface.co/datasets/ahigazy1/adrenal-pseudolabels . It was created and verified public; no inference masks have been produced yet. Existing private cache/model repositories keep their visibility. Only the pseudo-label runner explicitly opts into public output through `Hub(..., allow_public=True)`; training/cache callers retain their private-repository check.

You can change this same dataset to private later in Hugging Face settings. The runner and monitor accept either visibility for this output repository without changing its identity, prefix or progress. Authentication remains available through Secret Manager. Repository creation with exist_ok does not change an existing repository's visibility.

Destination checks: six monitor tests passed, pseudo-label regression checks passed, Bash syntax and git diff --check passed. The cache suite passed 27 tests; one pre-existing symlink test is blocked by Windows privilege error 1314. No training or inference VM was launched.
