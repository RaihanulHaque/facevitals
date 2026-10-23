"""Classical rPPG pulse-signal extraction and HR estimation.

All functions take an RGB trace of shape [N, 3] (per-frame ROI means, R,G,B order).
"""
from __future__ import annotations

import numpy as np
from scipy import signal
from scipy.ndimage import uniform_filter1d

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
    """CHROM (de Haan & Jeanne 2013).

    Full-length normalisation + band-pass of the X/Y chrominance signals (as in rPPG-Toolbox),
    then per-window alpha tuning with Hann overlap-add. Filtering inside 1.6 s windows (24 samples
    at 15 fps) is numerically unstable and was the source of a large upward HR bias.
    """
    n = len(rgb)
    L = max(int(win_sec * fps), 8)
    # running-mean normalisation over ~1 window, avoids the per-window DC step
    mean_rgb = uniform_filter1d(rgb, size=L, axis=0, mode="nearest")  # no zero-padding at the edges
    norm = rgb / (mean_rgb + 1e-8)
    xs = 3 * norm[:, 0] - 2 * norm[:, 1]
    ys = 1.5 * norm[:, 0] + norm[:, 1] - 1.5 * norm[:, 2]
    xf, yf = bandpass(xs, fps), bandpass(ys, fps)
    H = np.zeros(n)
    hann = np.hanning(L)
    for start in range(0, n - L + 1, L // 2):
        xw, yw = xf[start : start + L], yf[start : start + L]
        alpha = xw.std() / (yw.std() + 1e-8)
        H[start : start + L] += (xw - alpha * yw) * hann
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


def estimate_hr_fft(pulse: np.ndarray, fps: float, band=HR_BAND_HZ, nfft: int | None = None, harmonic: float = 0.5) -> float:
    """HR in bpm from the full-clip spectrum, harmonic-scored peak (see _band_peak)."""
    nfft = nfft or max(4096, 1 << int(np.ceil(np.log2(len(pulse) * 4))))
    f, p = spectrum_fft(pulse, fps, nfft)
    return _band_peak(f, p, band, harmonic)


def _band_peak(freqs: np.ndarray, power: np.ndarray, band=HR_BAND_HZ, harmonic: float = 0.0) -> float:
    """Peak frequency (bpm) inside the HR band.

    harmonic > 0 adds `harmonic * P(2f)` to the score: a real pulse has energy at its first
    harmonic, broadband motion/compression noise does not, so this favours the fundamental and
    suppresses the upward bias that plain argmax shows on low-SNR clips (noise argmax over
    42-180 bpm sits around 110 bpm).
    """
    m = (freqs >= band[0]) & (freqs <= band[1])
    if not m.any():
        return float("nan")
    fm, pm = freqs[m], power[m] / (power[m].max() + 1e-12)
    score = pm
    if harmonic:
        score = pm + harmonic * np.interp(2 * fm, freqs, power / (power[m].max() + 1e-12), right=0.0)
    return 60.0 * fm[np.argmax(score)]


def spectrum_fft(pulse: np.ndarray, fps: float, nfft: int = 1 << 16):
    """Full-clip Hann-windowed power spectrum. Maximum coherent gain for a stable pulse line;
    the only estimator that pulls a weak narrow pulse out of broadband motion noise."""
    x = pulse - pulse.mean()
    return np.fft.rfftfreq(nfft, 1 / fps), np.abs(np.fft.rfft(x * np.hanning(len(x)), nfft)) ** 2


def spectrum_welch(pulse: np.ndarray, fps: float, win_sec: float = 60.0, nfft: int = 1 << 16):
    """Welch power spectrum with long segments (60 s default). Tolerates HR drift on long clips.
    Segments shorter than ~30 s lose a weak pulse line to broadband noise at 15 fps."""
    x = pulse - pulse.mean()
    return signal.welch(x, fs=fps, nperseg=min(int(win_sec * fps), len(x)), nfft=nfft)


def _fuse(spectra: list[np.ndarray], freqs: np.ndarray, band=HR_BAND_HZ) -> np.ndarray:
    m = (freqs >= band[0]) & (freqs <= band[1])
    return sum(s / (s[m].sum() + 1e-12) for s in spectra)


def fused_hr_fft(pulses: list[np.ndarray], fps: float, band=HR_BAND_HZ, harmonic: float = 0.5) -> float:
    """Clip-level HR from several ROI pulse signals: in-band-normalised full-clip spectra summed."""
    specs = [spectrum_fft(p, fps) for p in pulses]
    f = specs[0][0]
    return _band_peak(f, _fuse([s for _, s in specs], f, band), band, harmonic)


def fused_hr_welch(pulses: list[np.ndarray], fps: float, band=HR_BAND_HZ, win_sec: float = 60.0, harmonic: float = 0.5) -> float:
    """Default clip-level estimator. harmonic=0.5 chosen by sweep on rppg_biases (88 subjects):
    0 -> +23 bpm bias (noise argmax), 1.0 -> -12 bpm bias (subharmonic trap), 0.5 -> -2 bpm."""
    specs = [spectrum_welch(p, fps, win_sec) for p in pulses]
    f = specs[0][0]
    return _band_peak(f, _fuse([s for _, s in specs], f, band), band, harmonic)


def fused_hr_windowed(pulses: list[np.ndarray], fps: float, band=HR_BAND_HZ, win_sec: float = 60.0,
                      step_sec: float = 10.0, harmonic: float = 0.5) -> float:
    """Median over long windows of the fused full-FFT harmonic-scored peak. Tolerates HR drift."""
    L, S = int(win_sec * fps), int(step_sec * fps)
    n = min(len(p) for p in pulses)
    if n < L:
        return fused_hr_fft([p[:n] for p in pulses], fps, band, harmonic)
    hrs = [fused_hr_fft([p[s0 : s0 + L] for p in pulses], fps, band, harmonic) for s0 in range(0, n - L + 1, S)]
    return float(np.median(hrs))


def estimate_hr_welch(pulse: np.ndarray, fps: float, band=HR_BAND_HZ, win_sec: float = 60.0, harmonic: float = 0.5) -> float:
    return fused_hr_welch([pulse], fps, band, win_sec, harmonic)


def hr_sliding_window(pulse: np.ndarray, fps: float, win_sec: float = 10.0, step_sec: float = 1.0, harmonic: float = 0.5):
    """Per-window HR estimates: returns (times_sec, hr_bpm)."""
    L, S = int(win_sec * fps), int(step_sec * fps)
    times, hrs = [], []
    for start in range(0, len(pulse) - L + 1, S):
        hrs.append(estimate_hr_fft(pulse[start : start + L], fps, harmonic=harmonic))
        times.append((start + L / 2) / fps)
    return np.array(times), np.array(hrs)


METHODS = {"green": green, "chrom": chrom, "pos": pos}
