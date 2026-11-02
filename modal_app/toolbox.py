"""Run rPPG-Toolbox (ubicomplab) on Modal against the facevitals-data volume.

    conda run -n ML modal run modal_app/toolbox.py --config UBFC-rPPG_UNSUPERVISED --dataset SCAMPS --data-path /data/scamps/example/scamps_videos_example
    conda run -n ML modal run modal_app/toolbox.py --config UBFC-rPPG_UNSUPERVISED --data-path /data/UBFC-rPPG/DATASET_2
    conda run -n ML modal run modal_app/toolbox.py --config SCAMPS_UBFC-rPPG_TSCAN_BASIC --data-path /data/UBFC-rPPG/DATASET_2 --gpu
    conda run -n ML modal run modal_app/toolbox.py --config SCAMPS_UBFC-rPPG_TSCAN_BASIC --data-path /data/UBFC-rPPG/DATASET_2 --gpu --model-path /data/toolbox_runs/<run>/PreTrainedModels/..pth

`--config` names a file in the toolbox's configs/infer_configs (or train_configs with --train). The runner
rewrites DATA_PATH / CACHED_PATH / LOG path into the volume, sets DO_PREPROCESS when the cache is empty,
and copies the toolbox's output CSVs + logs to /data/toolbox_runs/<config>_<tag>/.
Preprocessed chunks are cached on the volume at /data/toolbox_cache/<dataset> so re-runs are fast.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path

import modal

app = modal.App("facevitals-toolbox")
vol = modal.Volume.from_name("facevitals-data")

# Toolbox pins Python 3.8 + old scipy/numpy; we use 3.10 with relaxed pins and skip Mamba (PhysMamba only).
REQS = [
    "yacs==0.1.8", "scipy<1.12", "pandas<2.1", "scikit-image<0.22", "numpy<1.25", "matplotlib<3.8",
    "opencv-python-headless<4.9", "PyYAML==6.0", "scikit-learn<1.4", "tensorboardX==2.4.1", "tqdm",
    "mat73==0.59", "h5py<3.10", "fsspec", "timm==1.0.11", "protobuf==3.20.3", "neurokit2==0.2.10", "thop",
]
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "libgl1", "libglib2.0-0", "ffmpeg")
    .pip_install("torch==2.1.2", "torchvision==0.16.2", index_url="https://download.pytorch.org/whl/cu121")
    .pip_install(*REQS)
    .run_commands(
        "git clone --depth 1 https://github.com/ubicomplab/rPPG-Toolbox.git /app/rppg_toolbox",
        # PhysMamba imports mamba_ssm at module load; stub it so every other model works without the CUDA build.
        "python -c \"import site,os;p=os.path.join(site.getsitepackages()[0],'mamba_ssm');os.makedirs(p,exist_ok=True);open(os.path.join(p,'__init__.py'),'w').write('class Mamba:\\n    pass\\n')\"",
    )
)


@app.function(image=image, volumes={"/data": vol}, cpu=8, memory=32 * 1024, timeout=60 * 60 * 6)
def run_cpu(config: str, data_path: str, tag: str, train: bool, model_path: str | None, extra: dict) -> str:
    return _run(config, data_path, tag, train, model_path, extra)


@app.function(image=image, volumes={"/data": vol}, gpu="A10G", cpu=8, memory=32 * 1024, timeout=60 * 60 * 6)
def run_gpu(config: str, data_path: str, tag: str, train: bool, model_path: str | None, extra: dict) -> str:
    return _run(config, data_path, tag, train, model_path, extra)


def _run(config, data_path, tag, train, model_path, extra):
    import yaml

    tb = Path("/app/rppg_toolbox")
    sub = "train_configs" if train else "infer_configs"
    src = tb / "configs" / sub / f"{config}.yaml"
    if not src.exists():
        return f"no such config {src}; available: {sorted(p.stem for p in (tb / 'configs' / sub).glob('*.yaml'))}"
    cfg = yaml.safe_load(src.read_text())
    run_dir = Path(f"/data/toolbox_runs/{config}_{tag}")
    run_dir.mkdir(parents=True, exist_ok=True)

    def patch(section):
        d = cfg.get(section, {}).get("DATA")
        if not d:
            return
        if extra.get("dataset"):
            d["DATASET"] = extra["dataset"]
        ds = d["DATASET"]
        d["DATA_PATH"] = data_path if section in ("TEST", "UNSUPERVISED") or extra.get("train_path") is None else extra["train_path"]
        d["CACHED_PATH"] = f"/data/toolbox_cache/{ds}"
        Path(d["CACHED_PATH"]).mkdir(parents=True, exist_ok=True)
        d["DO_PREPROCESS"] = bool(extra.get("preprocess", True))  # toolbox caches per preprocessing variant; re-run unless told not to
        d["EXP_DATA_NAME"] = ""
        for k, v in extra.get("data", {}).items():
            d[k] = v

    for section in ("TRAIN", "VALID", "TEST", "UNSUPERVISED"):
        patch(section)
    cfg.setdefault("LOG", {})["PATH"] = str(run_dir / "runs")
    if model_path:
        cfg.setdefault("INFERENCE", {})["MODEL_PATH"] = model_path
    if "MODEL" in cfg:
        cfg["MODEL"]["MODEL_DIR"] = str(run_dir / "PreTrainedModels")
    cfg["DEVICE"] = "cuda:0" if extra.get("gpu") else "cpu"
    cfg["NUM_OF_GPU_TRAIN"] = 1 if extra.get("gpu") else 0
    cfg_path = run_dir / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    print(cfg_path.read_text(), flush=True)

    t0 = time.time()
    log = run_dir / "stdout.log"
    with open(log, "w") as lf:
        proc = subprocess.run(["python", "main.py", "--config_file", str(cfg_path)], cwd=tb, stdout=lf, stderr=subprocess.STDOUT, text=True)
    vol.commit()
    text = log.read_text()
    tail = "\n".join(text.splitlines()[-60:])
    return f"exit={proc.returncode} in {time.time()-t0:.0f}s\nrun_dir={run_dir}\n--- tail ---\n{tail}"


@app.local_entrypoint()
def main(config: str, data_path: str, tag: str = "run", gpu: bool = False, train: bool = False, model_path: str = None, train_path: str = None, fs: int = None, dataset: str = None, no_preprocess: bool = False):
    extra = {"train_path": train_path, "data": {}, "dataset": dataset, "preprocess": not no_preprocess, "gpu": gpu}
    if fs:
        extra["data"]["FS"] = fs
    fn = run_gpu if gpu else run_cpu
    print(fn.remote(config, data_path, tag, train, model_path, extra))
