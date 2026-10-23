#!/usr/bin/env python
"""Round 1 baseline: estimate HR from a facial video.

Usage:
    conda run -n ML python scripts/run_hr_video.py path/to/video.mp4 --method pos [--ref ref.csv] [--out results/out.csv]

--ref: optional CSV with columns time_sec,hr_bpm (ground truth) to compute the official metrics.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from facevitals.roi import extract_rgb_trace  # noqa: E402
from facevitals.rppg import METHODS, estimate_hr_welch, hr_sliding_window  # noqa: E402
from facevitals.metrics import evaluate_hr  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--method", choices=list(METHODS), default="pos")
    ap.add_argument("--ref", help="CSV with time_sec,hr_bpm ground truth")
    ap.add_argument("--out", default=None, help="CSV of per-window HR estimates")
    ap.add_argument("--win", type=float, default=10.0)
    ap.add_argument("--step", type=float, default=1.0)
    ap.add_argument("--max-frames", type=int, default=None)
    args = ap.parse_args()

    rgb, fps, _ = extract_rgb_trace(args.video, args.max_frames)
    print(f"frames={len(rgb)} fps={fps:.2f} duration={len(rgb)/fps:.1f}s")
    pulse = METHODS[args.method](rgb, fps)
    print(f"[{args.method}] clip HR (Welch 60 s, harmonic-scored) = {estimate_hr_welch(pulse, fps):.1f} bpm")

    t, hr = hr_sliding_window(pulse, fps, args.win, args.step)
    df = pd.DataFrame({"time_sec": t, "hr_bpm": hr})
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out, index=False)
        print(f"wrote {args.out}")

    if args.ref:
        ref = pd.read_csv(args.ref)
        ref_hr = np.interp(t, ref["time_sec"], ref["hr_bpm"])
        print(json.dumps(evaluate_hr(hr, ref_hr), indent=2))


if __name__ == "__main__":
    main()
