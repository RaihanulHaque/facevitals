"""Run scripts/eval_rppg_biases.py on Modal against the `facevitals-data` volume.

    conda run -n ML modal run modal_app/run_eval.py --roi forehead
Writes results CSV back to the volume at /data/results/ and prints the metrics table.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
app = modal.App("facevitals-eval")
vol = modal.Volume.from_name("facevitals-data")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("numpy", "scipy", "pandas", "opencv-python-headless")
    .add_local_dir(str(ROOT / "src"), remote_path="/app/src")
    .add_local_dir(str(ROOT / "scripts"), remote_path="/app/scripts")
)


@app.function(image=image, volumes={"/data": vol}, cpu=4, timeout=60 * 60)
def run(roi: str) -> str:
    out = f"/data/results/rppg_biases_{roi}.csv"
    Path("/data/results").mkdir(exist_ok=True)
    res = subprocess.run(
        ["python", "/app/scripts/eval_rppg_biases.py", "/data/rppg_biases", "--roi", roi, "--out", out],
        capture_output=True, text=True, cwd="/app",
    )
    vol.commit()
    tail = "\n".join(res.stdout.splitlines()[-12:])
    return tail + ("\nSTDERR:\n" + res.stderr[-2000:] if res.returncode else "")


@app.local_entrypoint()
def main(roi: str = "forehead"):
    print(run.remote(roi))
