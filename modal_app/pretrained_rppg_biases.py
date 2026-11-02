"""Run rPPG-Toolbox pretrained models (TS-CAN, EfficientPhys, PhysNet) on rppg_biases ROI videos.

    conda run -n ML modal run modal_app/pretrained_rppg_biases.py            # all models, 15 fps native
    conda run -n ML modal run modal_app/pretrained_rppg_biases.py --upsample  # 2x temporal interp -> 30 fps (models were trained at 30 fps)

The three ROI crops per subject are resized to 72x72 and preprocessed exactly like the toolbox
(DiffNormalized / Standardized), chunked (180 for TS-CAN/EfficientPhys, 128 for PhysNet), run through
the network, cumsum'd (outputs are diff-normalised pulse), band-passed, then HR = ROI-fused
harmonic-scored spectral peak (our facevitals estimator). Results -> /data/results/pretrained_rppg_biases*.csv
"""
from __future__ import annotations

import io
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
app = modal.App("facevitals-pretrained-rppg-biases")
vol = modal.Volume.from_name("facevitals-data")
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
        "python -c \"import site,os;p=os.path.join(site.getsitepackages()[0],'mamba_ssm');os.makedirs(p,exist_ok=True);open(os.path.join(p,'__init__.py'),'w').write('class Mamba:\\n    pass\\n')\"",
    )
    .add_local_dir(str(ROOT / "src"), remote_path="/app/src")
)

MODELS = {  # name -> (arch, weight file)
    "SCAMPS_TSCAN": ("tscan", "SCAMPS_TSCAN.pth"),
    "UBFC_TSCAN": ("tscan", "UBFC-rPPG_TSCAN.pth"),
    "PURE_TSCAN": ("tscan", "PURE_TSCAN.pth"),
    "UBFC_EfficientPhys": ("efficientphys", "UBFC-rPPG_EfficientPhys.pth"),
    "PURE_EfficientPhys": ("efficientphys", "PURE_EfficientPhys.pth"),
    "SCAMPS_PhysNet": ("physnet", "SCAMPS_PhysNet_DiffNormalized.pth"),
    "UBFC_PhysNet": ("physnet", "UBFC-rPPG_PhysNet_DiffNormalized.pth"),
    "PURE_PhysNet": ("physnet", "PURE_PhysNet_DiffNormalized.pth"),
}
ROIS = ["forehead", "leftcheek", "rightcheek"]


def _read_frames(path: str, upsample: bool):
    import cv2
    import numpy as np

    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2RGB), (72, 72), interpolation=cv2.INTER_AREA))
    cap.release()
    x = np.asarray(frames, dtype=np.float32)
    if upsample:  # linear temporal interpolation 15 -> 30 fps
        mid = 0.5 * (x[:-1] + x[1:])
        x = np.stack([x[:-1], mid], axis=1).reshape(-1, *x.shape[1:])
        fps *= 2
    return x, fps


def _diffnorm(data):
    import numpy as np

    d = (data[1:] - data[:-1]) / (data[1:] + data[:-1] + 1e-7)
    d = d / (np.std(d) + 1e-12)
    d = np.append(d, np.zeros((1,) + d.shape[1:], dtype=np.float32), axis=0)
    d[np.isnan(d)] = 0
    return d.astype(np.float32)


def _std(data):
    import numpy as np

    d = (data - np.mean(data)) / (np.std(data) + 1e-12)
    d[np.isnan(d)] = 0
    return d.astype(np.float32)


@app.function(image=image, volumes={"/data": vol}, gpu="A10G", cpu=4, memory=16 * 1024, timeout=60 * 60 * 3)
def run(upsample: bool, models: list[str]) -> str:
    import sys

    import numpy as np
    import pandas as pd
    import torch

    sys.path.insert(0, "/app/rppg_toolbox")
    sys.path.insert(0, "/app/src")
    from neural_methods.model.TS_CAN import TSCAN
    from neural_methods.model.EfficientPhys import EfficientPhys
    from neural_methods.model.PhysNet import PhysNet_padding_Encoder_Decoder_MAX
    from facevitals.rppg import bandpass, fused_hr_welch, fused_hr_fft
    from facevitals.metrics import evaluate_hr

    dev = "cuda"
    nets = {}
    for name in models:
        arch, wfile = MODELS[name]
        if arch == "tscan":
            m = TSCAN(frame_depth=10, img_size=72)
        elif arch == "efficientphys":
            m = EfficientPhys(frame_depth=10, img_size=72)
        else:
            m = PhysNet_padding_Encoder_Decoder_MAX(frames=128)
        sd = torch.load(f"/app/rppg_toolbox/final_model_release/{wfile}", map_location="cpu")
        sd = {k.replace("module.", ""): v for k, v in sd.items()}
        m.load_state_dict(sd)
        nets[name] = (arch, m.to(dev).eval())
        print("loaded", name, flush=True)

    @torch.no_grad()
    def predict(arch, net, frames):
        if arch == "physnet":
            L = 128
            dn = _diffnorm(frames)
            out = []
            for s in range(0, len(dn) - L + 1, L):
                x = torch.from_numpy(dn[s : s + L]).permute(3, 0, 1, 2)[None].contiguous().to(dev)  # [1,3,T,H,W]
                out.append(net(x)[0].squeeze().cpu().numpy())
            return np.concatenate(out) if out else np.zeros(0)
        L = 180
        n = (len(frames) // L) * L
        if arch == "tscan":
            x = np.concatenate([_diffnorm(frames[:n]), _std(frames[:n])], axis=-1)  # [n,72,72,6]
        else:
            x = _std(frames[:n])
        x = torch.from_numpy(x).permute(0, 3, 1, 2).contiguous()  # [n,C,H,W]
        n10 = (len(x) // 10) * 10
        x = x[:n10]
        if arch == "efficientphys":
            x = torch.cat([x, x[-1:]], 0)
        out = []
        for s in range(0, n10, 1800):
            xb = x[s : s + 1800 + (1 if arch == "efficientphys" else 0)].contiguous().to(dev)
            out.append(net(xb).squeeze().cpu().numpy())
        return np.concatenate(out)

    rows = []
    root = Path("/data/rppg_biases")
    for country in ["India", "Sierra_Leone"]:
        gt = pd.read_csv(root / "Ground_Truth" / f"{country}_GT.csv", encoding="utf-8-sig")
        gt.columns = [c.strip() for c in gt.columns]
        for _, r in gt.iterrows():
            vid = str(r.iloc[0]).strip()
            frames = {roi: _read_frames(str(root / "Data" / country / f"{vid}_{roi}.avi"), upsample) for roi in ROIS}
            fps = frames["forehead"][1]
            row = {"country": country, "video": vid, "ref_hr": float(r.iloc[1]), "fps": fps, "frames": len(frames["forehead"][0])}
            for name, (arch, net) in nets.items():
                pulses = []
                for roi in ROIS:
                    p = predict(arch, net, frames[roi][0])
                    if len(p) < 10 * fps:
                        continue
                    p = np.cumsum(p)  # diff-normalised output -> pulse
                    pulses.append(bandpass(p - p.mean(), fps))
                if pulses:
                    n = min(len(p) for p in pulses)
                    pulses = [p[:n] for p in pulses]
                    row[f"{name}_welch"] = fused_hr_welch(pulses, fps)
                    row[f"{name}_fft"] = fused_hr_fft(pulses, fps)
                else:
                    row[f"{name}_welch"] = row[f"{name}_fft"] = np.nan
            rows.append(row)
            print({k: (round(v, 1) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
    df = pd.DataFrame(rows)
    tag = "up30" if upsample else "fps15"
    Path("/data/results").mkdir(exist_ok=True)
    df.to_csv(f"/data/results/pretrained_rppg_biases_{tag}.csv", index=False)
    vol.commit()
    lines = [f"n={len(df)} upsample={upsample}"]
    for col in [c for c in df.columns if c.endswith(("_welch", "_fft"))]:
        m = evaluate_hr(df[col].values, df["ref_hr"].values)
        lines.append(f"{col:28s} MAE={m['MAE_bpm']:5.2f} RMSE={m['RMSE_bpm']:5.2f} r={m['pearson_r']:5.2f} bias={m['BA_bias_bpm']:6.2f}")
    return "\n".join(lines)


@app.local_entrypoint()
def main(upsample: bool = False, models: str = ",".join(MODELS)):
    print(run.remote(upsample, models.split(",")))
