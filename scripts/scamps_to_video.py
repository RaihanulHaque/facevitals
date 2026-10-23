#!/usr/bin/env python
"""Convert a SCAMPS .mat clip into an mp4 + reference HR csv for end-to-end pipeline tests.

    conda run -n ML python scripts/scamps_to_video.py data/raw/scamps/P000001.mat --out data/processed/scamps/P000001

Writes <out>.mp4 (RawFrames, 320x240, 30 fps, lossless-ish) and <out>_ref.csv (time_sec, hr_bpm) with
HR derived from the ground-truth PPG waveform `d_ppg` by peak detection in 10 s sliding windows, plus
<out>_ref_ppg.csv with the raw waveform. SCAMPS .mat files are v7.3 (HDF5): needs h5py.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

FPS = 30.0


def load_mat(path: Path):
    try:
        import h5py
    except ImportError as e:  # pragma: no cover
        raise SystemExit("SCAMPS .mat is v7.3/HDF5 -> `conda run -n ML pip install h5py`") from e
    with h5py.File(path, "r") as f:
        keys = list(f.keys())
        frames = np.array(f["RawFrames"])  # stored [t, h, w, c] or [c, w, h, t] depending on writer
        ppg = np.array(f["d_ppg"]).squeeze()
    return keys, frames, ppg


def frames_to_thwc(frames: np.ndarray) -> np.ndarray:
    # h5py returns MATLAB arrays transposed: MATLAB [c,w,h,t] -> numpy [t,h,w,c]
    if frames.shape[-1] == 3:
        return frames
    if frames.shape[0] == 3:
        return np.transpose(frames, (3, 2, 1, 0))
    raise ValueError(f"unexpected RawFrames shape {frames.shape}")


def ppg_to_hr(ppg: np.ndarray, fps: float, win: float = 10.0, step: float = 1.0):
    L, S = int(win * fps), int(step * fps)
    t, hr = [], []
    for s0 in range(0, len(ppg) - L + 1, S):
        seg = ppg[s0 : s0 + L]
        pk, _ = find_peaks(seg, distance=int(0.33 * fps))
        if len(pk) > 2:
            hr.append(60.0 * fps / np.mean(np.diff(pk)))
            t.append((s0 + L / 2) / fps)
    return np.array(t), np.array(hr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mat")
    ap.add_argument("--out", required=True, help="output prefix")
    args = ap.parse_args()
    keys, frames, ppg = load_mat(Path(args.mat))
    print("keys:", keys)
    frames = frames_to_thwc(frames)
    if frames.dtype != np.uint8:
        frames = np.clip(frames * (255.0 if frames.max() <= 1.0 else 1.0), 0, 255).astype(np.uint8)
    T, H, W, _ = frames.shape
    print(f"frames {frames.shape}, ppg {ppg.shape}")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(str(out.with_suffix(".mp4")), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    for fr in frames:
        vw.write(cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))
    vw.release()
    t, hr = ppg_to_hr(ppg, FPS)
    pd.DataFrame({"time_sec": t, "hr_bpm": hr}).to_csv(f"{out}_ref.csv", index=False)
    pd.DataFrame({"time_sec": np.arange(len(ppg)) / FPS, "ppg": ppg}).to_csv(f"{out}_ref_ppg.csv", index=False)
    print(f"wrote {out}.mp4 ({T} frames @ {FPS} fps), {out}_ref.csv (mean HR {hr.mean():.1f} bpm)")


if __name__ == "__main__":
    main()
