"""Official Task-1 evaluation metrics: MAE, RMSE, Pearson r, Bland-Altman limits of agreement."""
from __future__ import annotations

import numpy as np


def evaluate_hr(est: np.ndarray, ref: np.ndarray) -> dict:
    est, ref = np.asarray(est, float), np.asarray(ref, float)
    ok = ~(np.isnan(est) | np.isnan(ref))
    est, ref = est[ok], ref[ok]
    diff = est - ref
    r = float(np.corrcoef(est, ref)[0, 1]) if len(est) > 1 and est.std() > 0 and ref.std() > 0 else float("nan")
    bias = float(diff.mean())
    sd = float(diff.std(ddof=1)) if len(diff) > 1 else float("nan")
    return {
        "n": int(len(est)),
        "MAE_bpm": float(np.abs(diff).mean()),
        "RMSE_bpm": float(np.sqrt((diff**2).mean())),
        "pearson_r": r,
        "BA_bias_bpm": bias,
        "BA_LoA_lower_bpm": bias - 1.96 * sd,
        "BA_LoA_upper_bpm": bias + 1.96 * sd,
    }
