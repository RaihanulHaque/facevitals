#!/usr/bin/env python
"""Evaluate GREEN/CHROM/POS on the AiPEX-Lab rppg_biases dataset (pre-cropped ROI videos).

Expects the dataset root (local copy or Modal-mounted /data/rppg_biases):
    <root>/Data/{India,Sierra_Leone}/video<N>_{forehead,leftcheek,rightcheek}.avi
    <root>/Ground_Truth/{India_GT,Sierra_Leone_GT}.csv

Ground truth is the MEAN HR over the whole recording (15 fps, 30 s to 9 min), so the estimator is
clip-level: spectra of the three ROI pulse signals are normalised and summed, then peak-picked.
Estimators per method: plain full-clip FFT peak, and harmonic-scored (P(f)+P(2f)) peaks on the
full-clip FFT, on a 60 s Welch spectrum, and as the median over 60 s windows (drift-tolerant).

Usage:
    conda run -n ML python scripts/eval_rppg_biases.py data/raw/rppg_biases
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from facevitals.rppg import METHODS, fused_hr_fft, fused_hr_welch, fused_hr_windowed  # noqa: E402
from facevitals.metrics import evaluate_hr  # noqa: E402

ROIS = ["forehead", "leftcheek", "rightcheek"]


def video_mean_rgb(path: Path):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    rows = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        b, g, r = f.reshape(-1, 3).mean(axis=0)
        rows.append((r, g, b))
    cap.release()
    return np.array(rows), fps


def load_gt(root: Path, country: str) -> pd.DataFrame:
    df = pd.read_csv(root / "Ground_Truth" / f"{country}_GT.csv", encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--out", default="results/rppg_biases_eval.csv")
    ap.add_argument("--welch-win", type=float, default=60.0)
    args = ap.parse_args()
    root = Path(args.root)

    rows = []
    for country in ["India", "Sierra_Leone"]:
        gt = load_gt(root, country)
        id_col, hr_col = gt.columns[0], gt.columns[1]
        for _, r in gt.iterrows():
            vid = str(r[id_col]).strip()
            if not vid.startswith("video"):
                vid = f"video{int(vid)}"
            traces = {}
            for roi in ROIS:
                path = root / "Data" / country / f"{vid}_{roi}.avi"
                if path.exists():
                    rgb, fps = video_mean_rgb(path)
                    if len(rgb) >= 10 * fps:
                        traces[roi] = rgb
            if not traces:
                print(f"missing/short {country}/{vid}")
                continue
            n = min(len(t) for t in traces.values())
            row = {"country": country, "video": vid, "fps": fps, "frames": n, "n_rois": len(traces), "ref_hr": float(r[hr_col])}
            for name, fn in METHODS.items():
                pulses = [fn(t[:n], fps) for t in traces.values()]
                row[f"{name}_fft"] = fused_hr_fft(pulses, fps)
                row[f"{name}_fft_h"] = fused_hr_fft(pulses, fps, harmonic=0.5)
                row[f"{name}_welch_h"] = fused_hr_welch(pulses, fps, win_sec=args.welch_win, harmonic=0.5)
                row[f"{name}_win_h"] = fused_hr_windowed(pulses, fps, win_sec=args.welch_win, harmonic=0.5)
            rows.append(row)
            print({k: (round(v, 1) if isinstance(v, float) else v) for k, v in row.items()})

    df = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"\nwrote {args.out}  (n={len(df)})\n")
    est_cols = [c for c in df.columns if c.endswith(("_fft", "_fft_h", "_welch_h", "_win_h"))]
    df["const_mean"] = df["ref_hr"].mean()  # trivial baseline: GT spread is small on this dataset
    est_cols.append("const_mean")
    for col in est_cols:
        for country, sub in list(df.groupby("country")) + [("ALL", df)]:
            m = evaluate_hr(sub[col].values, sub["ref_hr"].values)
            print(f"{col:22s} {country:13s} n={m['n']:3d} MAE={m['MAE_bpm']:5.2f} RMSE={m['RMSE_bpm']:5.2f} r={m['pearson_r']:5.2f} "
                  f"bias={m['BA_bias_bpm']:6.2f} LoA=[{m['BA_LoA_lower_bpm']:6.1f},{m['BA_LoA_upper_bpm']:6.1f}]")


if __name__ == "__main__":
    main()
