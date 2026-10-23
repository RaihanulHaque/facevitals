# Resources (reviewed 2026-10-23)

All links below were checked. ✅ = reachable (HTTP 200). ⚠️ = blocked bot check (403) but the resource exists.

## Datasets
| Dataset | Vitals | Why it matters for us | Access | Link |
|---|---|---|---|---|
| UBFC-rPPG ✅ | HR, PPG (and SpO2 column in dataset 1 gtdump.xmp) | Listed in the official Annexure; the de-facto Round 1 benchmark. 8 + 42 videos, 640×480 @ 30 fps, uncompressed. | Request form (free, academic) | https://sites.google.com/view/ybenezeth/ubfcrppg |
| AiPEX-Lab rppg_biases ✅ | mean HR per video | **In the official Annexure.** 88 videos, India + Sierra Leone → directly serves **C2 skin tone**. Pre-cropped forehead/cheek ROIs. | Open (GitHub) | https://github.com/AiPEX-Lab/rppg_biases |
| PURE ✅ | HR, SpO2, PPG | 10 subjects, 6 motion conditions, 640×480 @ 30 fps. Clean HR + **SpO2** ground truth → Rounds 1–2. | Request via TU Ilmenau | (long TU Ilmenau URL, see teammate message) |
| UBFC-Phys ✅ | BVP, EDA | Stress tasks, 56 subjects. Good for motion/uncontrolled HR; no SpO2/BP. | Request form | https://sites.google.com/view/ybenezeth/ubfc-phys |
| iBVP ✅ | PPG, RGB + **thermal** | 32 subjects, labels for signal quality. Only RGB channel is allowed by the rules; ignore thermal. | Request (GitHub) | https://github.com/PhysiologicAILab/iBVP-Dataset |
| PhysDrive ✅ | HR, RR, SpO2?, in-vehicle | Harsh lighting / motion in cars → good stress test for **C1 low light**. Multi-modal; use RGB only. | Request (GitHub) | https://github.com/WJULYW/PhysDrive-Dataset |
| SUMS ✅ | HR, SpO2, RR, BP(?) | Multi-modal (RGB + NIR + wearable); check BP availability; useful for Round 3. | Request (GitHub) | https://github.com/thuhci/SUMS/ |
| FusionVitals ✅ | HR, SpO2, RR, **BP** | Code + dataset pointer for multi-vital estimation; the closest match to our Round 3 goal. | GitHub | https://github.com/McJackTang/FusionVitals/ |
| SCAMPS ✅ | HR, PPG, RR (synthetic) | 2,800 synthetic avatar videos with **controllable skin tone and lighting** → cheap pre-training and C1/C2 ablations. | Open download | https://github.com/danmcduff/scampsdataset |
| MMPD ✅ | HR, PPG | 33 subjects, **Fitzpatrick 3–6**, 4 lighting levels incl. low light, 4 motion types. Best public match for **C1 + C2 + C4** (320×240 @ 30 fps). | Request (GitHub) | https://github.com/THU-CS-PI/MMPD_rPPG_dataset |
| COHFACE ⚠️ (Zenodo 403 to bots) | HR, BVP, RR | 40 subjects, studio + natural lighting, heavily MPEG-4 compressed → serves **C5 low bitrate**. | Zenodo / Idiap EULA | https://zenodo.org/records/4081054 |
| BU-3DFE ✅ | none (facial expressions) | 3D expression database, **no physiological ground truth**. Only useful for face-landmark / pose experiments. Low priority. | Request | https://www.cs.binghamton.edu/~lijun/Research/3DFE/3DFE_Analysis.html |

Not in the teammate list but worth knowing: **VitalVideos** (900 subjects, HR+BP, diverse skin tones; paid/institutional), **V4V** (HR+BP from video, CVPR challenge), **BP4D+** (BP, HR, RR; request), **MTHS** (smartphone HR+SpO2, open).

## Toolbox
- **rPPG-Toolbox** ✅ https://github.com/ubicomplab/rPPG-Toolbox — paper https://arxiv.org/pdf/2210.00716
  Unsupervised baselines (GREEN, ICA, POS, CHROM, LGI, PBV, OMIT) + supervised (DeepPhys, TS-CAN, EfficientPhys, PhysNet, PhysFormer, iBVPNet, RhythmFormer, FactorizePhys) with loaders for UBFC-rPPG, PURE, SCAMPS, MMPD, UBFC-Phys, iBVP, BP4D+. Also supports **cross-dataset eval** and **low-res / compression augmentations**. Plan: use it as the evaluation harness rather than re-implement loaders.

## Round 1 — HR
- Paper ✅ https://ieeexplore.ieee.org/document/10214238 (behind IEEE paywall for full text).
- Code: abhisheksrivastava99 ✅, TomChow01 ✅, habom2310 ✅ — all three are classical OpenCV + FFT/ICA student projects (Haar/dlib face → ROI mean → bandpass → FFT peak). Fine as reference; our `src/facevitals/` already implements a cleaner version of this (GREEN / CHROM / POS).

## Round 2 — BP (the teammate labeled this Round 2, but per the PDF **Round 2 is HR + SpO2** and BP comes in **Round 3**)
- MDPI Sensors 2023 ⚠️ https://www.mdpi.com/1424-8220/23/6/2963 — "Remote BP estimation via spatiotemporal mapping of facial videos" (MDPI blocks bots; opens in a browser).
- PMC ✅ https://pmc.ncbi.nlm.nih.gov/articles/PMC11781723/
- Code ✅ https://github.com/AiPEX-Lab/vbpe — video BP estimation from the same lab as `rppg_biases`.

## Round 3 — SpO2 (per the PDF this is actually **Round 2** material)
- Paper ✅ https://arxiv.org/abs/2212.07116 — smartphone-camera SpO2 (contact PPG from finger over camera; adaptation to face needs ratio-of-ratios on red/blue AC/DC of facial rPPG).
- Code ✅ poorvabedmutha31 (smartphone finger PPG) and Si-tong-416 (small personal repo). Both are starting points only; neither is face-based.

## Takeaways
1. Register the team by **25 Oct 2026** — nothing else matters until that's done.
2. Request **UBFC-rPPG, PURE, MMPD** now (approval takes days to weeks). Clone `rppg_biases` and `SCAMPS` immediately (open).
3. Use **rPPG-Toolbox** as the evaluation backbone; keep our own lightweight pipeline for the Raspberry Pi real-time demo.
4. Cheapest constraints to address convincingly for Round 1: **C4 (low-res)** and **C5 (low fps / bitrate)** — both are pure simulations on any dataset. **C2** is covered by `rppg_biases` + MMPD.
