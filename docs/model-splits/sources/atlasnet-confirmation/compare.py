import gzip, json, zipfile
from pathlib import Path
import nibabel as nib, numpy as np
from scipy.signal import fftconvolve
want = json.load(open('/tmp/atlasnet_matched_scans.json')); D = Path('/opt/adrenal-multiorgan/data')
def ours(key):
    src, name = key.split('/', 1)
    if src == 'ts':
        return nib.Nifti1Image.from_bytes(gzip.decompress(zipfile.ZipFile(D / 'sources/Totalsegmentator_dataset_v201.zip').read(f'{name}/ct.nii.gz')))
    for p in {'amos': ['amos/train/imagesTr', 'amos/valid/imagesVa'], 'btcv': ['btcv/RawData/Training/img'], 'flare': ['flare/images']}[src]:
        if (D / p / name).exists(): return nib.load(D / p / name)
load = lambda img: np.clip(np.asanyarray(nib.as_closest_canonical(img).dataobj).astype(np.float32), -1000, 1000)
for bd, key in sorted(want.items()):
    A, O = load(nib.load(Path('/opt/atlas-check', bd, 'ct.nii.gz'))), load(ours(key))
    if A.shape[2] != O.shape[2] or A.shape[0] > O.shape[0] or A.shape[1] > O.shape[1]:
        print(f'{bd} {key:34s} not a crop: atlas {A.shape} ours {O.shape}', flush=True); continue
    z = A.shape[2] // 2
    a, o = A[:, :, z] - A[:, :, z].mean(), O[:, :, z]
    score = fftconvolve(o - o.mean(), a[::-1, ::-1], mode='valid')  # best in-plane offset of the crop
    x0, y0 = np.unravel_index(np.argmax(score), score.shape)
    block = O[x0:x0 + A.shape[0], y0:y0 + A.shape[1], :]
    r = np.corrcoef(block.ravel()[::7], A.ravel()[::7])[0, 1]
    print(f'{bd} {key:34s} offset ({x0},{y0}) 3D correlation {r:.5f}  median |diff| {np.median(np.abs(block - A)):.1f} HU', flush=True)
