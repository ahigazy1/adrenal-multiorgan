"""Efficient per-shard AbdomenAtlas overlap matcher.
Reuses cached gold standardization (.atlas_std_cache.pkl) -> only standardizes the
new shard's ct.nii.gz (native disk, fast). scipy.ndimage.label(26-conn) == skimage
bounds (validated) but C-fast. Appends matches to atlas_overlap.csv with a shard col.

usage: python .atlas_match3.py <shard_tag> <ct_glob>
"""
import os, glob, time, hashlib, pickle, sys, csv
import numpy as np, nibabel as nib
from nibabel.orientations import io_orientation, apply_orientation
from scipy import ndimage as ndi
from scipy.ndimage import zoom
from multiprocessing import Pool

GOLD = "/mnt/c/Users/ahiga/Desktop/AdrenalGold"
NWORK = 10
STRUCT = np.ones((3, 3, 3), bool)   # 26-conn == AbdomenAtlas skimage default

def standardize(path):
    try:
        img = nib.load(path); data = np.array(img.dataobj)
        if data.ndim != 3: data = data.squeeze()
        data[data > 1000] = 1000; data[data < -1000] = -1000
        cond = (data > -100) & (data < 100)
        lab, n = ndi.label(cond, structure=STRUCT)
        if n == 0: return (path, None, None, None)
        counts = np.bincount(lab.ravel()); counts[0] = 0
        largest = int(counts.argmax())
        sl = ndi.find_objects(lab)[largest - 1]
        crop = data[sl]
        ras = apply_orientation(crop, io_orientation(img.affine)).astype(np.int16)
        shp = tuple(int(s) for s in ras.shape)
        md5 = hashlib.md5(np.ascontiguousarray(ras)).hexdigest()
        w = np.clip(ras.astype(np.float32), -200, 300)
        d = zoom(w, tuple(32.0 / s for s in w.shape), order=1)[:32, :32, :32]
        if d.shape != (32, 32, 32):
            d = np.pad(d, [(0, 32 - d.shape[i]) for i in range(3)])
        v = d.ravel().astype(np.float32); v -= v.mean(); s = v.std()
        if s > 0: v /= s
        return (path, shp, md5, v)
    except Exception:
        return (path, None, None, None)

if __name__ == "__main__":
    shard_tag, ct_glob = sys.argv[1], sys.argv[2]
    cache = pickle.load(open(f"{GOLD}/.atlas_std_cache.pkl", "rb"))
    G = [r for r in cache if r[1] != "ATLAS" and r[2] is not None]   # (path, ds, shape, md5, v)
    g_md5 = {}
    for path, ds, shp, md5, v in G:
        g_md5.setdefault(md5, []).append((ds, path, shp))
    GV = np.stack([r[4] for r in G]); N = GV.shape[1]
    gmeta = [(r[1], r[0], r[2]) for r in G]   # ds, path, shape

    cts = sorted(glob.glob(ct_glob, recursive=True))
    print(f"{shard_tag}: {len(cts)} atlas ct | gold cached {len(G)} | {NWORK} workers", flush=True)
    t0 = time.time()
    with Pool(NWORK) as pool:
        res = pool.map(standardize, cts, chunksize=4)
    ok = [r for r in res if r[1] is not None]
    print(f"standardized {len(ok)}/{len(res)} in {time.time()-t0:.0f}s", flush=True)
    pickle.dump(res, open(f"{GOLD}/.atlas_std_{shard_tag}.pkl", "wb"))

    def gname(ds, gp):
        return os.path.basename(os.path.dirname(gp)) if ds == "RAOS" else os.path.basename(gp)

    rows = []
    for path, shp, md5, v in ok:
        bd = path.split("/")[-2]
        if md5 in g_md5:
            for ds, gp, gshp in g_md5[md5]:
                rows.append((shard_tag, ds, gname(ds, gp), bd, "exact", "1.0000"))
            continue
        corr = (GV @ v) / N; j = int(np.argmax(corr)); bestc = float(corr[j])
        ds, gp, gshp = gmeta[j]
        if bestc > 0.97 and gshp == shp:
            rows.append((shard_tag, ds, gname(ds, gp), bd, "shape+corr", f"{bestc:.4f}"))
        elif bestc > 0.99:
            rows.append((shard_tag, ds, gname(ds, gp), bd, "corr-only", f"{bestc:.4f}"))

    master = f"{GOLD}/atlas_overlap.csv"
    header = ["shard", "gold_dataset", "gold_scan", "atlas_bdmap_id", "match_type", "corr32"]
    existing = []
    if os.path.exists(master):
        rd = list(csv.reader(open(master)))
        if rd:
            existing = [r for r in rd[1:] if r and r[0] != shard_tag]   # drop this shard's old rows (idempotent)
    with open(master, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(existing)
        w.writerows(sorted(rows))
    from collections import Counter
    print(f"{shard_tag}: {len(rows)} matches appended -> {master}", flush=True)
    print("  by dataset:", dict(Counter(r[1] for r in rows)), flush=True)
    print("  by type:", dict(Counter(r[4] for r in rows)), flush=True)
