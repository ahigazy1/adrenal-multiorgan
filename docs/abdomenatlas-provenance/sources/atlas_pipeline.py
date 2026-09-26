#!/usr/bin/env python3
"""Robust, resumable AbdomenAtlas shard 02-05 overlap pipeline (pure Python, no xet).
Per shard: requests-stream the tarball (Range-resume) -> extract only ct.nii.gz ->
run .atlas_match3.py (idempotent CSV) -> write .matched marker -> drop extracted ct.
Re-run safe: done shards skipped, partial .part resumes."""
import os, glob, time, tarfile, subprocess, shutil
import requests
from huggingface_hub import get_token

GOLD = "/mnt/c/Users/ahiga/Desktop/AdrenalGold"
DATA = "/home/ahigazy/data"
PY   = "/home/ahigazy/cu13test/.venv/bin/python"
REPO = "BodyMaps/_AbdomenAtlas1.1Mini"
SHARDS = {"06": "00002501_00003000", "07": "00003001_00003500",
          "08": "00003501_00004000", "09": "00004001_00004500"}
TOKEN = get_token()
HEADERS = {"Authorization": f"Bearer {TOKEN}"} if TOKEN else {}
LOG = f"{GOLD}/.atlas_pipeline.log"

def log(m):
    line = f"[{time.strftime('%m-%d %H:%M:%S')}] {m}"
    print(line, flush=True)
    try:
        with open(LOG, "a") as f: f.write(line + "\n")
    except OSError: pass

def total_size(url):
    try:
        r = requests.head(url, headers=HEADERS, allow_redirects=True, timeout=60)
        for k in ("Content-Length", "x-linked-size", "X-Linked-Size"):
            v = r.headers.get(k)
            if v and int(v) > 0: return int(v)
    except Exception as e:
        log(f"  head error: {str(e)[:80]}")
    return 0

def download(url, dest):
    part = dest + ".part"
    total = total_size(url)
    log(f"  total size {total/1e9:.2f} GB")
    last_log = os.path.getsize(part) if os.path.exists(part) else 0
    for attempt in range(400):
        existing = os.path.getsize(part) if os.path.exists(part) else 0
        if total and existing >= total: break
        hh = dict(HEADERS)
        if existing: hh["Range"] = f"bytes={existing}-"
        try:
            with requests.get(url, headers=hh, stream=True, timeout=(30, 180)) as r:
                r.raise_for_status()
                done = existing
                with open(part, "ab" if existing else "wb") as f:
                    for chunk in r.iter_content(8 * 1024 * 1024):
                        if chunk:
                            f.write(chunk); done += len(chunk)
                            if done - last_log >= 2_000_000_000:
                                log(f"  {done/1e9:.1f}/{total/1e9:.1f} GB"); last_log = done
        except Exception as e:
            log(f"  retry {attempt} (have {os.path.getsize(part) if os.path.exists(part) else 0:_} B): {str(e)[:80]}")
            time.sleep(5)
    final = os.path.getsize(part) if os.path.exists(part) else 0
    if total and final >= total:
        os.replace(part, dest); return True
    if not total and final > 0:
        os.replace(part, dest); return True   # unknown size, single-shot completed
    return False

def main():
    log("=== python pipeline start (requests, no xet) ===")
    if not TOKEN: log("WARNING: no HF token found")
    for s, rng in SHARDS.items():
        marker = f"{DATA}/.shard{s}.matched"
        if os.path.exists(marker):
            log(f"shard{s}: already matched, skip"); continue
        d = f"{DATA}/abdomen_atlas_shard{s}"; os.makedirs(d, exist_ok=True)
        tb = f"AbdomenAtlas1.1Mini_BDMAP_{rng}.tar.gz"
        dest = f"{d}/{tb}"
        url = f"https://huggingface.co/datasets/{REPO}/resolve/main/{tb}"
        if not os.path.exists(dest):
            log(f"shard{s}: downloading {tb}")
            if not download(url, dest):
                log(f"shard{s}: download incomplete, will retry next run"); continue
        log(f"shard{s}: have tarball {os.path.getsize(dest)/1e9:.2f} GB; extracting ct.nii.gz")
        try:
            n = 0
            with tarfile.open(dest, "r:gz") as tf:
                for m in tf:
                    if m.name.endswith("ct.nii.gz"):
                        tf.extract(m, d); n += 1
            log(f"shard{s}: extracted {n} ct.nii.gz")
        except Exception as e:
            log(f"shard{s}: extract FAILED {str(e)[:100]}; will retry"); continue
        ct_glob = f"{d}/**/ct.nii.gz"
        log(f"shard{s}: matching ({len(glob.glob(ct_glob, recursive=True))} ct)")
        rc = subprocess.run([PY, f"{GOLD}/.atlas_match3.py", f"shard{s}", ct_glob]).returncode
        if rc == 0:
            open(marker, "w").close()
            shutil.rmtree(d, ignore_errors=True)   # drop tarball+extracted to bound disk
            log(f"shard{s}: DONE")
        else:
            log(f"shard{s}: matcher rc={rc}, will retry next run")
    done = sum(os.path.exists(f"{DATA}/.shard{s}.matched") for s in SHARDS)
    log(f"=== pipeline finished: {done}/4 shards matched ===")

main()
