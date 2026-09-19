# Preprocessing cache and fresh-VM recovery

`gcp/startup.sh` runs `pixi run python hf_cache.py ensure --data data` before training. `ensure` makes the
prepared data available on the local disk and uploads nothing. The upload is a second command,
`hf_cache.py publish`, which `startup.sh` starts in the background at the lowest CPU and disk priority, so
the GPU never waits for a 170 GB transfer that only a future replacement machine needs.
The default cache is the private dataset repository `ahigazy1/adrenal-multiorgan-cache`.
Checkpoints remain in the private model repository `ahigazy1/adrenal-multiorgan-model`.
`ADRENAL_CACHE_REPO` and `ADRENAL_HF_REPO` can override those destinations.

## First successful preparation

When there is no complete compatible cache, the VM runs `prepare.sh`.
Preprocessing uses 24 workers, with raw tqdm output redirected to a file.
After preparation succeeds, the cache validator requires all development cases,
five nonoverlapping train/validation splits, foreground-sampling index files,
validation ground truth, the held-out test images and labels, and the checksum-verified
AtlasNet weights. It includes the generated plans and dataset/build metadata.

It does not upload the source archives, raw training CTs, credentials, temporary files,
training results or preprocessing logs as part of this cache. The prepared training
arrays, case properties and sampling store are sufficient for training; native-space
test images and labels are retained for the final evaluation.

Files are checksummed and packed into roughly 10 GiB uncompressed tar archives. Only
the tar archives and a completion manifest are uploaded, avoiding thousands of Hub files.
Each archive is uploaded and deleted locally before building the next, so there is no
second 200 GB archive or full staging copy. Completed archive uploads can be reused
after an interruption on the same disk. A local `.preprocessed-local.json` records the
completed build before uploading so retrying an upload does not rerun preprocessing.

A `complete.json` manifest is committed last, only after the remote payload has been
verified. The local `.prepared` receipt is written once the local data is complete and its
manifest is built (files are hashed with 16 threads), or after a verified restore; it does not
wait for the upload. Training never starts with incomplete local data. On a restart of the same
machine, `ensure` checks that every expected file is present but does not hash 170 GB again, and
`publish` continues with the parts that are not on Hugging Face yet (log:
`/var/log/adrenal-cache-upload.log`). `startup.sh` waits for the upload at the very end, after
prediction, before the machine switches itself off.

## New VM or deleted disk

A fresh VM computes a recipe fingerprint from the preparation scripts, source-download
pins, plans, `pixi.lock` and vendored nnU-Net Python sources. A complete cache for that
exact recipe is downloaded from a single pinned Hugging Face commit. Every downloaded
archive is checksum-verified before extraction, then every extracted file is verified. Source downloads, cleaning, splitting,
resampling and sampling-index extraction are skipped on a cache hit.

A partial remote upload is not a complete cache. If its VM and disk were deleted before
publication, another VM must prepare the data again (about 40 minutes). Deletion cannot recover data that
never reached Hugging Face. No cache from the failed September 19 preprocessing run is
created retroactively by this code change.

Missing/changed files, bad credentials, permission errors and network failures stop the
job with an error; they are not treated as permission to silently build or overwrite a
new experiment. Public artifact repositories are rejected. Changes to the recipe cause
a cache miss, and a checkpoint from a different recipe or data payload cannot be resumed.

## Training checkpoints

Every 100 epochs, the trainer publishes the checkpoints, run record, settings, logs and
`_hf_snapshot.json` together using one explicit atomic Hub commit. A failed upload leaves
the previous complete snapshot authoritative. No training hyperparameters were changed.

On a fresh disk, `train.py` verifies the remote run record against the local trainer,
plans, split and preprocessing content ID. It stages and verifies the entire snapshot
before installing its model directory. A partial download cannot become a local
checkpoint. Only files listed in the snapshot are restored, so stale checkpoint files
left over in the remote repository cannot override the current snapshot.

Resume uses nnU-Net's full checkpoint loader (`continue_training=True`), including its
saved optimizer/scaler state and epoch, not the initial-weights loading path. This is
not a guarantee of bitwise-identical random augmentation after a restart. Recovery is
from the last successfully uploaded checkpoint; later work on a deleted disk is lost.
Legacy remote model folders without a snapshot manifest are rejected rather than
silently replaced. A matching final checkpoint plus completed validation skips training
and proceeds to the test-set prediction step.

## Permissions and launch

The Secret Manager secret `HF_TOKEN` must grant **read and write access to both repos**.
A token with read-only access to the cache (the previous requirement) is insufficient
for publishing a prepared cache. Grant that permission or replace the secret before
launching. The token is not stored in GitHub or in the artifact payload.

From an existing clone in Google Cloud Shell:

```bash
git pull --ff-only && bash gcp/create_vm.sh
```

The first new VM builds the cache and publishes it while it trains. VMs created after that upload has completed restore it automatically.
If no training checkpoint has ever been uploaded, training starts from AtlasNet rather
than inventing an earlier training state.

## Checks

```bash
python -m pytest tests
bash -n gcp/startup.sh
python -m py_compile hf_cache.py train.py trainers/adrenal_multiorgan.py
```

The tests use a fake, revisioned Hub and tiny synthetic files. They cover missing files,
incorrect splits, incomplete sampling indexes, interrupted batch uploads, completion
markers, corruption, pinned downloads, path traversal, cache reuse, snapshot atomicity
and checkpoint compatibility. They do not replace a full GCE/Hugging Face transfer or
GPU training test with the real dataset.
