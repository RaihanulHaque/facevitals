#!/usr/bin/env python
"""Sweep clip-level HR estimators on cached rppg_biases traces (no video decoding).

    conda run -n ML python scripts/sweep_estimators.py data/processed/rppg_biases_traces.npz

Reports MAE / bias / r for each (method, spectrum, harmonic weight, subharmonic guard, band) combo
against the dataset's mean-HR ground truth, plus the constant-mean baseline for reference.
"""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from facevitals.rppg import METHODS, spectrum_fft, spectrum_welch, _fuse  # noqa: E402
from facevitals.metrics import evaluate_hr  # noqa: E402

ROIS = ["forehead", "leftcheek", "rightcheek"]


def load(npz_path: Path, gt_dir: Path):
    z = np.load(npz_path)
    fps = float(z["fps"][0])
    subjects = []
    for country in ["India", "Sierra_Leone"]:
        gt = pd.read_csv(gt_dir / f"{country}_GT.csv", encoding="utf-8-sig")
        gt.columns = [c.strip() for c in gt.columns]
        for _, r in gt.iterrows():
            vid = str(r.iloc[0]).strip()
            traces = [z[f"{country}__{vid}_{roi}"] for roi in ROIS if f"{country}__{vid}_{roi}" in z]
            if traces:
                n = min(len(t) for t in traces)
                subjects.append((country, vid, float(r.iloc[1]), [t[:n] for t in traces]))
    return fps, subjects


def pick(f, P, lo, hi, harmonic, guard):
    m = (f >= lo) & (f <= hi)
    fm, pm = f[m], P[m] / (P[m].max() + 1e-12)
    Pn = P / (P[m].max() + 1e-12)
    score = pm + harmonic * np.interp(2 * fm, f, Pn, right=0.0)
    if guard:
        # subharmonic guard: penalise f if P(f/2) is comparable (then f is probably a harmonic),
        # but also penalise f if P(2f) is much larger than P(f) (then f is probably a subharmonic)
        sub = np.interp(fm / 2, f, Pn, left=0.0)
        score = score - guard * np.maximum(0, sub - pm)
    return 60.0 * fm[np.argmax(score)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--gt", default="data/raw/rppg_biases/Ground_Truth")
    ap.add_argument("--out", default="results/estimator_sweep.csv")
    args = ap.parse_args()
    fps, subjects = load(Path(args.npz), Path(args.gt))
    print(f"fps={fps} subjects={len(subjects)}")
    ref = np.array([s[2] for s in subjects])

    # precompute fused spectra per (method, spectrum type)
    spectra = {}
    for name, fn in METHODS.items():
        for stype, sfn in [("fft", lambda p: spectrum_fft(p, fps)), ("welch60", lambda p: spectrum_welch(p, fps, 60))]:
            per_subject = []
            for _, _, _, traces in subjects:
                specs = [sfn(fn(t, fps)) for t in traces]
                f = specs[0][0]
                per_subject.append((f, _fuse([s for _, s in specs], f)))
            spectra[(name, stype)] = per_subject
            print(f"spectra ready: {name}/{stype}")

    rows = []
    grid = itertools.product(METHODS, ["fft", "welch60"], [0.0, 0.25, 0.5, 1.0], [0.0, 0.5, 1.0], [(0.7, 3.0), (0.7, 2.5)])
    for name, stype, harm, guard, (lo, hi) in grid:
        est = np.array([pick(f, P, lo, hi, harm, guard) for f, P in spectra[(name, stype)]])
        m = evaluate_hr(est, ref)
        rows.append({"method": name, "spectrum": stype, "harmonic": harm, "guard": guard, "band_hi_bpm": hi * 60, **m})
    m = evaluate_hr(np.full_like(ref, ref.mean()), ref)
    rows.append({"method": "const_mean", "spectrum": "-", "harmonic": 0, "guard": 0, "band_hi_bpm": 0, **m})
    df = pd.DataFrame(rows).sort_values("MAE_bpm")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    pd.set_option("display.width", 200)
    print(df.head(25).round(2).to_string(index=False))
    print("...")
    print(df[df.method == "const_mean"].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
