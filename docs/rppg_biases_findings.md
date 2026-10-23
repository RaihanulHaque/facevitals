# rppg_biases baseline: what we learned (2026-10-23)

Dataset: AiPEX-Lab `rppg_biases` (official Annexure). 88 subjects (44 India, 44 Sierra Leone), three
pre-cropped ROI videos each (forehead 60×30, cheeks 25×25, upscaled to 200×200), MPEG-4, **15 fps**,
30 s to 11 min. Ground truth = one **mean HR per recording** from a Masimo pulse oximeter.

## Headline numbers (88 subjects, ROI-fused spectra)
| Estimator | MAE | bias | r |
|---|---|---|---|
| constant = dataset mean HR (77 bpm) | **4.5** | 0.0 | 0.00 |
| green, Welch 60 s, harmonic 0.5 (**current default**) | 11.6 | −2.0 | 0.13 |
| pos, Welch 60 s, harmonic 0.25 | 12.3 | −1.5 | 0.12 |
| green, Welch 60 s, harmonic 1.0 | 12.6 | −11.8 | −0.05 |
| green, full-clip FFT, plain argmax (first attempt) | 25.7 | +23.0 | 0.03 |
| authors' CHROM / POS (from `Results_Final_Main.csv`) | 10–17 | −11…+13 | ≈0 |
| authors' best (BKF) | 4–6 | ≈0 | ≈0 |

Reference HR has SD 5.5 bpm (66–93), so a constant guess scores MAE 4.5. **Nobody, including the
authors, gets per-subject correlation above ~0.2 on this data.** The paper's "~1 bpm" is population
mean bias, not per-subject error.

## Why it was biased and what fixed it
1. **fps is right (15).** Not a metadata problem; confirmed with ffprobe and spectral checks.
2. **Signal is weak.** After band-passing, R/G/B are 0.99 correlated → dominant modulation is
   luminance (tracker jitter + compression), not blood. CHROM/POS cancel it and are left with noise;
   CHROM locks at ~62 bpm on many clips (likely MPEG-4 GOP periodicity, 1 Hz at 15 fps).
3. **Short 10 s windows at 15 fps** → frequency resolution 6 bpm and SNR too low; argmax follows
   noise. Noise argmax over 42–180 bpm averages ~110 bpm → the +20 bpm bias.
4. **Harmonic-scored peak picking** (`score = P(f) + w·P(2f)`): w=1 overshoots into subharmonics
   (−12 bpm), **w=0.5 removes the bias**. Set as default in `fused_hr_*` and `estimate_hr_welch`.
5. **ROI fusion** (sum of in-band-normalised spectra from forehead + both cheeks) helps on the short
   clips; long clips stay noise-limited.
6. Original CHROM implementation band-passed inside 1.6 s windows (24 samples) — numerically bad.
   Rewritten to filter full-length X/Y (as rPPG-Toolbox does).

## Implications for the competition
- rppg_biases is a **fairness/bias** benchmark (per-country / skin-tone bias), not an accuracy
  benchmark. Use it for the C2 report (bias per stratum), not for headline MAE.
- Headline HR accuracy must come from UBFC-rPPG / PURE / MMPD (full-face, uncompressed/30 fps).
- A learned prior (BKF-style Bayesian tracking) is what made the authors' best method look good;
  worth adding as a post-processor once we have per-window estimates on better data.
- Our pipeline must never ship plain argmax on the 42–180 band; keep harmonic scoring + long windows.

## Reproduce
```bash
conda run -n ML modal run modal_app/extract_traces.py          # cache traces (~1 min)
conda run -n ML modal volume get facevitals-data /cache/rppg_biases_traces.npz data/processed/
conda run -n ML python scripts/sweep_estimators.py data/processed/rppg_biases_traces.npz
```
