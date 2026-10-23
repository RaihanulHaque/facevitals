"""Extract per-frame mean RGB traces for every rppg_biases ROI video once, cache to the volume.

    conda run -n ML modal run modal_app/extract_traces.py
Output: /data/cache/rppg_biases_traces.npz  (key "<country>/<video>/<roi>" -> [N,3] float32 RGB, plus "fps")
Afterwards:  modal volume get facevitals-data /cache/rppg_biases_traces.npz data/processed/
"""
from __future__ import annotations

import io
from pathlib import Path

import modal

app = modal.App("facevitals-extract-traces")
vol = modal.Volume.from_name("facevitals-data")
image = modal.Image.debian_slim(python_version="3.12").pip_install("numpy", "opencv-python-headless")


@app.function(image=image, volumes={"/data": vol}, cpu=2, timeout=60 * 30)
def trace_one(rel: str) -> tuple[str, bytes, float]:
    import cv2
    import numpy as np

    cap = cv2.VideoCapture(f"/data/rppg_biases/Data/{rel}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    rows = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        b, g, r = f.reshape(-1, 3).mean(axis=0)
        rows.append((r, g, b))
    cap.release()
    buf = io.BytesIO()
    np.save(buf, np.asarray(rows, dtype=np.float32))
    return rel, buf.getvalue(), fps


@app.function(image=image, volumes={"/data": vol}, timeout=60 * 60)
def run() -> str:
    import numpy as np

    rels = sorted(str(p.relative_to("/data/rppg_biases/Data")) for p in Path("/data/rppg_biases/Data").rglob("*.avi"))
    out, fps_all = {}, {}
    for rel, blob, fps in trace_one.map(rels):
        key = rel.replace(".avi", "").replace("_", "/", 1) if False else rel[:-4]
        out[key] = np.load(io.BytesIO(blob))
        fps_all[key] = fps
    Path("/data/cache").mkdir(exist_ok=True)
    np.savez_compressed("/data/cache/rppg_biases_traces.npz", fps=np.array(list(fps_all.values())), keys=np.array(list(out.keys())), **{k.replace("/", "__"): v for k, v in out.items()})
    vol.commit()
    return f"{len(out)} traces cached; fps set={sorted(set(fps_all.values()))}"


@app.local_entrypoint()
def main():
    print(run.remote())
