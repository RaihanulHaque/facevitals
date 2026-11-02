"""Stream the first N clips out of the 600 GB SCAMPS tarball into the volume (no full download).

    conda run -n ML modal run modal_app/scamps_slice.py --n 300

The tar is read sequentially over HTTP; we stop after N .mat members, so only ~N x 120 MB is
transferred. Written to /data/scamps/videos/PXXXXXX.mat. Idempotent: skips if N already present.
"""
from __future__ import annotations

import io
import os
import tarfile
import time
import urllib.request
from pathlib import Path

import modal

app = modal.App("facevitals-scamps-slice")
vol = modal.Volume.from_name("facevitals-data")
image = modal.Image.debian_slim(python_version="3.12")
URL = "https://facesyntheticspubwedata.z6.web.core.windows.net/neurips-2022/scamps_videos.tar.gz"
DEST = "/data/scamps/videos"


@app.function(image=image, volumes={"/data": vol}, timeout=60 * 60 * 6)
def stream_slice(n: int) -> str:
    Path(DEST).mkdir(parents=True, exist_ok=True)
    have = sorted(p.name for p in Path(DEST).glob("*.mat"))
    if len(have) >= n:
        return f"already have {len(have)} clips"
    t0 = time.time()
    req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
    resp = urllib.request.urlopen(req, timeout=120)
    tar = tarfile.open(fileobj=resp, mode="r|gz")
    got, seen = len(have), 0
    for member in tar:
        if not member.isfile() or not member.name.endswith(".mat"):
            continue
        seen += 1
        name = os.path.basename(member.name)
        if name in have:
            continue
        f = tar.extractfile(member)
        out = Path(DEST) / name
        with open(out.with_suffix(".part"), "wb") as w:
            while True:
                chunk = f.read(8 << 20)
                if not chunk:
                    break
                w.write(chunk)
        os.rename(out.with_suffix(".part"), out)
        got += 1
        if got % 10 == 0:
            vol.commit()
            print(f"{got}/{n} clips, {time.time()-t0:.0f}s, {sum(p.stat().st_size for p in Path(DEST).glob('*.mat'))/1e9:.1f} GB", flush=True)
        if got >= n:
            break
    vol.commit()
    return f"{got} clips in {DEST} after {time.time()-t0:.0f}s"


@app.local_entrypoint()
def main(n: int = 300):
    print(stream_slice.remote(n))
