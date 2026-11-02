"""Download UBFC-rPPG (public Google Drive folder, no request form) into the volume.

    conda run -n ML modal run modal_app/fetch_ubfc.py
Layout on volume: /data/UBFC-rPPG/DATASET_1/<name>/{vid.avi,gtdump.xmp}, /data/UBFC-rPPG/DATASET_2/subjectN/{vid.avi,ground_truth.txt}
(= the layout rPPG-Toolbox's UBFC-rPPG loader expects for DATASET_2). Resumable: existing files are skipped.
Cite: Bobbia et al., "Unsupervised skin tissue segmentation for remote photoplethysmography", PRL 2017.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import modal

app = modal.App("facevitals-fetch-ubfc")
vol = modal.Volume.from_name("facevitals-data")
image = modal.Image.debian_slim(python_version="3.12").pip_install("gdown")
FOLDER = "https://drive.google.com/drive/folders/1o0XU4gTIo46YfwaWjIgbtCncc-oF44Xk"
DEST = "/data/UBFC-rPPG"


@app.function(image=image, volumes={"/data": vol}, timeout=60 * 60 * 6)
def fetch() -> str:
    import gdown

    t0 = time.time()
    files = gdown.download_folder(FOLDER, skip_download=True, quiet=True)
    print(f"{len(files)} files in Drive folder", flush=True)
    done, skipped, failed = 0, 0, []
    for i, f in enumerate(files):
        out = Path(DEST) / f.path
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.exists() and out.stat().st_size > 0:
            skipped += 1
            continue
        for attempt in range(3):
            try:
                gdown.download(id=f.id, output=str(out) + ".part", quiet=True)
                os.rename(str(out) + ".part", out)
                done += 1
                break
            except Exception as e:  # rate limits etc.
                print(f"retry {attempt} {f.path}: {e}", flush=True)
                time.sleep(20 * (attempt + 1))
        else:
            failed.append(f.path)
        if (done + skipped) % 10 == 0:
            vol.commit()
            print(f"{done + skipped}/{len(files)} ({time.time()-t0:.0f}s)", flush=True)
    vol.commit()
    size = sum(p.stat().st_size for p in Path(DEST).rglob("*") if p.is_file()) / 1e9
    return f"done={done} skipped={skipped} failed={failed} total={size:.1f} GB in {time.time()-t0:.0f}s"


@app.local_entrypoint()
def main():
    print(fetch.remote())
