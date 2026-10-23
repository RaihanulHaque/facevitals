"""Classical rPPG pulse-signal extraction and HR estimation.

All functions take an RGB trace of shape [N, 3] (per-frame ROI means, R,G,B order).
"""
from __future__ import annotations

import numpy as np
from scipy import signal

HR_BAND_HZ = (0.7, 3.0)  # 42-180 bpm, as in the official spec


def _detrend(x: np.ndarray, fps: float, cutoff_hz: float = 0.5) -> np.ndarray:
    """Remove slow illumination drift with a high-pass Butterworth filter."""
    b, a = signal.butter(2, cutoff_hz / (fps / 2), btype="high")
    return signal.filtfilt(b, a, x, axis=0)


def bandpass(x: np.ndarray, fps: float, band=HR_BAND_HZ, order: int = 3) -> np.ndarray:
    lo, hi = band
    hi = min(hi, 0.99 * fps / 2)  # guard for low-fps (C5) input
    b, a = signal.butter(order, [lo / (fps / 2), hi / (fps / 2)], btype="band")
    return signal.filtfilt(b, a, x, axis=0)


def green(rgb: np.ndarray, fps: float) -> np.ndarray:
    """Simplest baseline: green channel only (Verkruysse et al. 2008)."""
    return bandpass(_detrend(rgb[:, 1], fps), fps)


def chrom(rgb: np.ndarray, fps: float, win_sec: float = 1.6) -> np.ndarray:
    """CHROM (de Haan & Jeanne 2013) with overlap-add windows."""
    n = len(rgb)
    L = int(win_sec * fps)
    H = np.zeros(n)
    for start in range(0, n - L + 1, L // 2):
        seg = rgb[start : start + L]
        norm = seg / (seg.mean(axis=0) + 1e-8)
        xs = 3 * norm[:, 0] - 2 * norm[:, 1]
        ys = 1.5 * norm[:, 0] + norm[:, 1] - 1.5 * norm[:, 2]
        xf, yf = bandpass(xs, fps), bandpass(ys, fps)
        alpha = xf.std() / (yf.std() + 1e-8)
        h = xf - alpha * yf
        H[start : start + L] += h * np.hanning(L)
    return H


def pos(rgb: np.ndarray, fps: float, win_sec: float = 1.6) -> np.ndarray:
    """POS (Wang et al. 2017)."""
    n = len(rgb)
    L = int(win_sec * fps)
    H = np.zeros(n)
    P = np.array([[0, 1, -1], [-2, 1, 1]], dtype=float)
    for end in range(L, n + 1):
        seg = rgb[end - L : end]
        Cn = seg / (seg.mean(axis=0) + 1e-8)
        S = Cn @ P.T  # [L, 2]
        h = S[:, 0] + (S[:, 0].std() / (S[:, 1].std() + 1e-8)) * S[:, 1]
        H[end - L : end] += h - h.mean()
    return bandpass(H, fps)


def estimate_hr_fft(pulse: np.ndarray, fps: float, band=HR_BAND_HZ, nfft: int | None = None) -> float:
    """Return HR in bpm from the dominant spectral peak inside the HR band."""
    x = pulse - pulse.mean()
    nfft = nfft or max(4096, 1 << int(np.ceil(np.log2(len(x) * 4))))
    freqs = np.fft.rfftfreq(nfft, d=1 / fps)
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x)), n=nfft))
    mask = (freqs >= band[0]) & (freqs <= band[1])
    if not mask.any():
        return float("nan")
    return 60.0 * freqs[mask][np.argmax(spec[mask])]


def hr_sliding_window(pulse: np.ndarray, fps: float, win_sec: float = 10.0, step_sec: float = 1.0):
    """Per-window HR estimates: returns (times_sec, hr_bpm)."""
    L, S = int(win_sec * fps), int(step_sec * fps)
    times, hrs = [], []
    for start in range(0, len(pulse) - L + 1, S):
        hrs.append(estimate_hr_fft(pulse[start : start + L], fps))
        times.append((start + L / 2) / fps)
    return np.array(times), np.array(hrs)


METHODS = {"green": green, "chrom": chrom, "pos": pos}
