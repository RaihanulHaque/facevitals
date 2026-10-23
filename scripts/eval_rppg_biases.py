#!/usr/bin/env python
"""Evaluate GREEN/CHROM/POS on the AiPEX-Lab rppg_biases dataset (pre-cropped ROI videos).

Expects the dataset root (local copy or Modal-mounted /data/rppg_biases):
    <root>/Data/{India,Sierra_Leone}/video<N>_{forehead,leftcheek,rightcheek}.avi
    <root>/Ground_Truth/{India_GT,Sierra_Leone_GT}.csv

Usage:
    conda run -n ML python scripts/eval_rppg_biases.py data/raw/rppg_biases --roi forehead
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from facevitals.rppg import METHODS, hr_sliding_window  # noqa: E402
from facevitals.metrics import evaluate_hr  # noqa: E402


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
    ap.add_argument("--roi", choices=["forehead", "leftcheek", "rightcheek"], default="forehead")
    ap.add_argument("--out", default="results/rppg_biases_eval.csv")
    ap.add_argument("--win", type=float, default=10.0, help="window length (s) for HR estimates")
    ap.add_argument("--step", type=float, default=2.0)
    args = ap.parse_args()
    root = Path(args.root)

    rows = []
    for country in ["India", "Sierra_Leone"]:
        gt = load_gt(root, country)
        print(f"{country}: GT columns = {list(gt.columns)}")
        # first column = video id/name, second = mean HR (verify once data is present)
        id_col, hr_col = gt.columns[0], gt.columns[1]
        for _, r in gt.iterrows():
            vid = str(r[id_col]).strip()
            if not vid.startswith("video"):
                vid = f"video{int(vid)}"
            path = root / "Data" / country / f"{vid}_{args.roi}.avi"
            if not path.exists():
                print(f"missing {path}")
                continue
            rgb, fps = video_mean_rgb(path)
            if len(rgb) < 5 * fps:
                continue
            row = {"country": country, "video": vid, "fps": fps, "frames": len(rgb), "ref_hr": float(r[hr_col])}
            for name, fn in METHODS.items():
                # long clips (up to ~9 min @ 15 fps): median of 10 s window estimates, not one global FFT peak
                _, hr_w = hr_sliding_window(fn(rgb, fps), fps, win_sec=args.win, step_sec=args.step)
                row[f"hr_{name}"] = float(np.nanmedian(hr_w)) if len(hr_w) else float("nan")
            rows.append(row)
            print(row)

    df = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"\nwrote {args.out}  (n={len(df)})\n")
    for name in METHODS:
        for country, sub in list(df.groupby("country")) + [("ALL", df)]:
            m = evaluate_hr(sub[f"hr_{name}"].values, sub["ref_hr"].values)
            print(f"{name:6s} {country:13s} n={m['n']:3d} MAE={m['MAE_bpm']:5.2f} RMSE={m['RMSE_bpm']:5.2f} r={m['pearson_r']:5.2f} "
                  f"BA=[{m['BA_LoA_lower_bpm']:6.1f},{m['BA_LoA_upper_bpm']:6.1f}]")


if __name__ == "__main__":
    main()
