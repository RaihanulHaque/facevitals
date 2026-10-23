#!/usr/bin/env python
"""Sanity check the rPPG pipeline on a synthetic RGB trace with a known pulse (no video needed)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from facevitals.rppg import METHODS, estimate_hr_fft, hr_sliding_window  # noqa: E402
from facevitals.metrics import evaluate_hr  # noqa: E402

rng = np.random.default_rng(0)
fps, dur, hr_bpm = 30.0, 30.0, 72.0
t = np.arange(0, dur, 1 / fps)
pulse = np.sin(2 * np.pi * hr_bpm / 60 * t)
drift = 0.03 * np.sin(2 * np.pi * 0.05 * t)
base = np.array([150.0, 110.0, 90.0])
amp = np.array([0.4, 0.8, 0.3])  # green carries the strongest pulsatile component
rgb = base[None, :] * (1 + drift[:, None]) + amp[None, :] * pulse[:, None] + rng.normal(0, 0.5, (len(t), 3))

for name, fn in METHODS.items():
    sig = fn(rgb, fps)
    hr_est = estimate_hr_fft(sig, fps)
    tt, hh = hr_sliding_window(sig, fps)
    m = evaluate_hr(hh, np.full_like(hh, hr_bpm))
    print(f"{name:6s} clip HR={hr_est:6.1f} bpm (true {hr_bpm})  windowed MAE={m['MAE_bpm']:.2f} RMSE={m['RMSE_bpm']:.2f}")
