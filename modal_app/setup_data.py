"""Populate the `facevitals-data` Modal Volume with the open datasets.

Run (personal profile `raihanulhaque007` must be active):
    conda run -n ML modal run modal_app/setup_data.py            # everything
    conda run -n ML modal run modal_app/setup_data.py --only rppg_biases
    conda run -n ML modal run modal_app/setup_data.py --only scamps --scamps-full   # 600 GB, opt-in

Volume layout:
    /data/rppg_biases/        git clone of AiPEX-Lab/rppg_biases (88 subjects x 3 ROI videos + GT csv)
    /data/scamps/example/     10 example .mat videos (1.2 GB)
    /data/scamps/waveforms/   labels-only csv for all 2,800 videos (60 MB) + train/val/test split
    /data/scamps/full/        only if --scamps-full
    /data/MANIFEST.json       what was fetched, when, sizes
Datasets that need a signed request (UBFC-rPPG, PURE, MMPD, COHFACE) are uploaded later with
`modal volume put facevitals-data <local> /<name>` once approved.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import modal

VOL_NAME = "facevitals-data"
MOUNT = "/data"

app = modal.App("facevitals-setup-data")
vol = modal.Volume.from_name(VOL_NAME, create_if_missing=True)
image = modal.Image.debian_slim(python_version="3.12").apt_install("git", "curl", "aria2")

SCAMPS_BASE = "https://facesyntheticspubwedata.z6.web.core.windows.net/neurips-2022"
SCAMPS_FILES = {
    "example": ["scamps_videos_example.tar.gz"],
    "waveforms": ["scamps_waveforms_csv.tar.gz", "ScampsTrainTestSplit.csv"],
    "full": ["scamps_videos.tar.gz"],
}


def _sh(cmd: str, cwd: str | None = None) -> None:
    print(f"$ {cmd}", flush=True)
    subprocess.run(cmd, shell=True, check=True, cwd=cwd)


def _du(path: str) -> int:
    return sum(p.stat().st_size for p in Path(path).rglob("*") if p.is_file())


def _fetch(url: str, dest_dir: str) -> str:
    Path(dest_dir).mkdir(parents=True, exist_ok=True)
    name = url.rsplit("/", 1)[-1]
    out = os.path.join(dest_dir, name)
    if os.path.exists(out):
        print(f"skip (exists) {out}")
        return out
    _sh(f"aria2c -x 8 -s 8 --file-allocation=none -d {dest_dir} -o {name}.part '{url}'")
    os.rename(f"{out}.part", out)
    return out


@app.function(image=image, volumes={MOUNT: vol}, timeout=60 * 60 * 6)
def fetch_rppg_biases() -> dict:
    dest = f"{MOUNT}/rppg_biases"
    if os.path.exists(f"{dest}/Ground_Truth/India_GT.csv") and len(list(Path(f"{dest}/Data").rglob("*.avi"))) >= 264:
        print("rppg_biases already present")
    else:
        _sh(f"rm -rf {dest} && git clone --depth 1 https://github.com/AiPEX-Lab/rppg_biases.git {dest}")
        _sh(f"rm -rf {dest}/.git")
    vol.commit()
    n = len(list(Path(f"{dest}/Data").rglob("*.avi")))
    return {"name": "rppg_biases", "videos": n, "bytes": _du(dest)}


@app.function(image=image, volumes={MOUNT: vol}, timeout=60 * 60 * 12)
def fetch_scamps(part: str) -> dict:
    dest = f"{MOUNT}/scamps/{part}"
    for fname in SCAMPS_FILES[part]:
        path = _fetch(f"{SCAMPS_BASE}/{fname}", dest)
        if fname.endswith(".tar.gz"):
            marker = path + ".extracted"
            if not os.path.exists(marker):
                _sh(f"tar -xzf {path} -C {dest}")
                Path(marker).touch()
                os.remove(path)  # keep the volume lean
        vol.commit()
    return {"name": f"scamps/{part}", "files": len(list(Path(dest).rglob("*"))), "bytes": _du(dest)}


@app.function(image=image, volumes={MOUNT: vol})
def write_manifest(entries: list[dict]) -> str:
    man_path = f"{MOUNT}/MANIFEST.json"
    manifest = json.load(open(man_path)) if os.path.exists(man_path) else {"datasets": {}}
    for e in entries:
        manifest["datasets"][e["name"]] = {**e, "updated": datetime.now(timezone.utc).isoformat()}
    json.dump(manifest, open(man_path, "w"), indent=2)
    vol.commit()
    return json.dumps(manifest, indent=2)


@app.function(image=image, volumes={MOUNT: vol})
def tree(depth: int = 2) -> str:
    out = subprocess.run(f"cd {MOUNT} && find . -maxdepth {depth} | sort | head -80 && echo && du -sh */ 2>/dev/null", shell=True, capture_output=True, text=True)
    return out.stdout


@app.local_entrypoint()
def main(only: str = "all", scamps_full: bool = False):
    t0 = time.time()
    jobs = []
    if only in ("all", "rppg_biases"):
        jobs.append(fetch_rppg_biases.spawn())
    if only in ("all", "scamps"):
        jobs.append(fetch_scamps.spawn("waveforms"))
        jobs.append(fetch_scamps.spawn("example"))
        if scamps_full:
            jobs.append(fetch_scamps.spawn("full"))
    results = [j.get() for j in jobs]
    for r in results:
        print(f"{r['name']:<20} {r.get('videos', r.get('files')):>6} items  {r['bytes']/1e9:6.2f} GB")
    print(write_manifest.remote(results))
    print(tree.remote())
    print(f"done in {time.time()-t0:.0f}s")
