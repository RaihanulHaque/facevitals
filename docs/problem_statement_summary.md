# SP Cup 2027 — FaceVitals: Problem Statement Summary

Source: `docs/SP-Cup-2027-FaceVitals-Official.pdf` (18 pages, IEEE SPS, sponsored by MathWorks).
Official page: https://2027.ieeeicassp.org/sp-cup/

## The task in one line
Build a **real-time** system that estimates **Heart Rate (HR), SpO2, and Blood Pressure (BP)** from
**RGB-only facial video** (no NIR, thermal, or contact sensors), that works **offline and on live streams**,
on **low-resolution input**, and is **clinically validated** against reference devices / standard datasets.
Must generalize across **skin tone, gender, facial orientation**, and work at **≤ 50 lux**.
Edge deployment (Raspberry Pi) is recommended; MATLAB Raspberry Pi Blockset is the suggested common target.

## Key dates
| Milestone | Date |
|---|---|
| Team registration deadline | **25 Oct 2026** (2 days from today, 23 Oct 2026) |
| Training & starter code brainstorming session | TBD |
| Round 1 submission | **30 Nov 2026** |
| Round 1 results | 5 Dec 2026 |
| Round 2 submission | **16 Jan 2027** |
| Round 2 results | 21 Jan 2027 |
| Finalist selection | 21 Jan – 1 Feb 2027 |
| 3 finalists announced | 9 Feb 2027 |
| Final at ICASSP 2027 | 16–21 May 2027 |

Registration: https://www2.securecms.com/SPCup/SPCRegistration.asp
All members must sign the SP/VIP Cup Terms & Conditions before the registration deadline.

## Rounds
| Round | Scope | Notes |
|---|---|---|
| Task 1 / Round 1 | **HR only**, controlled + uncontrolled conditions | Must address and justify **≥ 2 constraints (C1–C7)** even at baseline. Judged on innovation, feasibility, documentation clarity. Least deviation from medical reference gets priority. |
| Round 2 | **HR + SpO2** jointly | Quantitative comparison vs ground-truth pulse oximeter required. |
| Round 3 | **HR + SpO2 + BP**, full system | Clinical validation mandatory (standard dataset and/or clinical reference device). **Live/real-time demo under ≥ 2 constraints.** Judged on per-parameter accuracy, robustness, real-time perf (Raspberry Pi). |

## Task 1 baseline specification (ideal case)
- Indoor, **≥ 300 lux**, static non-reflective background, minimal motion.
- Smartphone-grade RGB camera / USB webcam, **≥ 1280×720**, **≥ 30 fps**, 40–60 cm, frontal ±10°, uncompressed or H.264 ≥ 5 Mbps.
- Single subject; ROI from facial landmarks (68 or 468 pt mesh) → **forehead + bilateral cheeks**; no mask/glare/facial hair on ROI.
- Ground truth: clinical pulse oximeter or ECG, timestamp-synchronized, ≥ 100 Hz for ECG.
- **Metrics: MAE (bpm), RMSE (bpm), Pearson r, Bland–Altman limits of agreement.**

## Constraints C1–C7 (each has a quantified spec)
| ID | Constraint | Quantified spec | Suggested simulation |
|---|---|---|---|
| C1 | Low light | **≤ 50 lux** at the face | brightness scaling + Poisson/Gaussian noise injection |
| C2 | Skin tone & dermatological conditions | Fitzpatrick I–VI (verify via **ITA angle** from L\*a\*b\*); per-type error must stay within tolerance of population mean; handle vitiligo | `rgb2lab` → ITA stratification, report per-stratum metrics |
| C3 | Multi-face tracking | Detect all faces, lock onto one designated target, reject bystanders | face detector + Kalman tracking + KLT / optical flow |
| C4 | Low resolution | **≤ 640×480 (VGA)**, stress case **≤ 320×240 (QVGA)**; baseline 1280×720 | bicubic downsample then re-run pipeline |
| C5 | Low bandwidth | **≤ 500 kbps**, **≤ 10 fps** (Nyquist floor ≈ 6 fps for 0.7–3 Hz HR band) | frame decimation + MPEG-4 re-encode at constrained bitrate |
| C6 | Aging skin | 60–70+ yrs; report accuracy separately for this subgroup | edge-emphasizing `imfilter` micro-texture perturbation (synthetic only) |
| C7 | Masked / occluded | **≥ 40 % of lower face occluded**; re-weight toward forehead/periocular | synthetic occlusion patch aligned to landmarks |

FAQ: you may address any subset; **more constraints addressed → counts toward grading**. You may build your own supplementary dataset (needs IRB/ethics clearance + consent, documents verified).

## Submission format (strict)
Technical report in **IEEE conference paper format** + code. Files per round:
- Round 1: `Round1_Baseline_None.mat`, `process_round1.m` (MATLAB **or Python** script), `target_video.mp4`, `reference_signal.csv`, `processed_output.csv`
- Round 2: `Round2_[Condition]_[Constraint].mat`, `process_round2.m`, same files + SpO2 channels
- Round 3: real-time deployable code, `target_video.mp4` (or live-stream log), `reference_signal.csv` (HR, SpO2, BP), `processed_output.csv`, `benchmark_comparison.csv`

Each `.mat` must contain: `facial_video`, `roi_data`, `reference_signal`, `processed_signal`, `metrics`, `params`
(`params` = resolution, fps, lux, distance, Fitzpatrick type, age, constraints applied).

Directory structure: `/Round1_Baseline/`, `/Round2_HR_SpO2/`, `/Round3_Constrained/`, `/Round4_FullSystem_Clinical/`
File naming (own data): `Subject{ID}_{SkinTone}_{Condition}_{Parameter}.ext`, e.g. `Subject014_TypeV_LowLight_HR.csv`
Condition tags: Controlled, Hospital, LowLight, LowRes. Suggested: `/Dataset/Controlled/`, `/Dataset/Hospital/`, `/Dataset/LowLight/`.

## Ground-truth instruments
HR → pulse oximeter or ECG · SpO2 → clinical pulse oximeter · BP → calibrated oscillometric / auscultatory sphygmomanometer.

## Data options (Section 3)
1. Own dataset with IRB clearance (documents submitted in Round 1 judging).
2. Public datasets (Annexure 1 lists UBFC-rPPG and AiPEX-Lab `rppg_biases`, 88 videos from India + Sierra Leone with forehead/cheek ROIs and mean-HR CSVs).
3. Organizer dataset from Sri Ramachandra Hospital, Tamil Nadu (IEC/23/FEB/177/08) — requires a copyright/legal agreement; contact Dr. Vijay Jeyakumar (vijayjeyakumar@ieee.org, cc sp-competitions@listserv.ieee.org).

## Team rules
- 1 faculty supervisor + **3–10 undergraduates** (+ optional 1 graduate tutor). ≥ 3 undergrads must be SPS student members at registration through the end.
- Members frozen after registration deadline. One finalist per institution.
- ≥ 1 undergrad must attend ICASSP in person for the final (travel grants: $1,200 continental / $1,700 intercontinental for up to 3 students; free registration for up to 3).
- Prizes: $5,000 / $2,500 / $1,500.
- Ineligible: ≥ 50 % of undergrads were on a finalist team last year; anyone on a top-3 team at the previous SPS competition.

## Final judging (six equally weighted criteria)
Innovation & thoroughness · validation-set performance · test-set performance · model & time complexity · report quality · presentation quality.

## Note
Submitted work may be made available to MathWorks for commercial use.
