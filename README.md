# FaceVitals — IEEE SP Cup 2027

Contactless, bias-robust estimation of **Heart Rate, SpO2 and Blood Pressure** from RGB facial video.

- Problem statement digest: [docs/problem_statement_summary.md](docs/problem_statement_summary.md)
- Official PDF: [docs/SP-Cup-2027-FaceVitals-Official.pdf](docs/SP-Cup-2027-FaceVitals-Official.pdf)
- Datasets, toolboxes, papers (reviewed): [docs/resources.md](docs/resources.md)

## Deadlines
Registration **25 Oct 2026** · Round 1 (HR) **30 Nov 2026** · Round 2 (HR+SpO2) **16 Jan 2027** · Finalists 9 Feb 2027 · Final at ICASSP 16–21 May 2027.

## Environment
Everything Python runs in the conda env **ML**:
```bash
conda run -n ML python scripts/synthetic_check.py
```
```bash
conda run -n ML python scripts/run_hr_video.py data/raw/some_video.mp4 --method pos --out results/hr.csv
```

## Data on Modal
Datasets live in Modal Volume **`facevitals-data`** (personal workspace `raihanulhaque007`), mounted at `/data`. See [modal_app/README.md](modal_app/README.md).
```bash
conda run -n ML modal run modal_app/setup_data.py
```

## Face detector
OpenCV 5 (in env ML) has no Haar cascades; the pipeline uses **YuNet** (`models/face_detection_yunet_2023mar.onnx`, from opencv_zoo, 230 KB, committed). Gives a box + 5 landmarks → forehead / cheek ROIs.

## End-to-end test on a synthetic SCAMPS clip
```bash
conda run -n ML python scripts/scamps_to_video.py data/raw/scamps/P000001.mat --out data/processed/scamps/P000001
```
```bash
conda run -n ML python scripts/run_hr_video.py data/processed/scamps/P000001.avi --method pos --ref data/processed/scamps/P000001_ref.csv
```

## Live webcam demo (Mac)
```bash
conda run -n ML python scripts/webcam_hr.py --method green
```
```bash
conda run -n ML python scripts/webcam_hr.py --method pos --model models/trained/tscan_scamps300_ep5.pth
```
Keys: `q` quit, `r` reset, `1/2/3` = green/chrom/pos. HR appears after 12 s of face. Good light, sit still, 40–60 cm. First run asks for camera permission for your terminal app.
The `.pth` is not in git (download: `conda run -n ML modal volume get facevitals-data /toolbox_runs/SCAMPS_SCAMPS_UBFC-rPPG_TSCAN_BASIC_scamps300/PreTrainedModels/SCAMPS_SCAMPS_UBFC_tscan_Epoch4.pth models/trained/tscan_scamps300_ep5.pth`).

## Training / evaluation with rPPG-Toolbox on Modal
`modal_app/toolbox.py` runs the toolbox against the volume (unsupervised, pretrained `only_test`, or `train_and_test`). See its docstring. Results and weights land in `/data/toolbox_runs/<config>_<tag>/`.
Datasets on the volume: `/data/scamps/videos` (300 clips), `/data/scamps/example` (10), `/data/rppg_biases`, `/data/UBFC-rPPG` (partial; Drive quota, resumable with `modal_app/fetch_ubfc.py`).
Dataset access status + request email drafts: [docs/dataset_requests.md](docs/dataset_requests.md).

## rppg_biases findings
See [docs/rppg_biases_findings.md](docs/rppg_biases_findings.md): dataset is SNR-limited (authors' own r≈0); our default estimator is now green + 60 s Welch + harmonic-scored peak (bias −2 bpm, was +23).

## Layout
```
src/facevitals/   roi.py (YuNet face ROI), rppg.py (GREEN/CHROM/POS + harmonic HR), metrics.py, models/ts_can.py (vendored, RAIL license)
scripts/          run_hr_video.py (Round 1 CLI), webcam_hr.py (live demo), synthetic_check.py, eval_rppg_biases.py, sweep_estimators.py, scamps_to_video.py
models/           YuNet face-detector ONNX
modal_app/        setup_data.py, fetch_ubfc.py, scamps_slice.py, extract_traces.py, run_eval.py, toolbox.py (rPPG-Toolbox runner), pretrained_rppg_biases.py
docs/             problem statement summary, resources, official PDF
data/raw          put datasets here (git-ignored)
results/          outputs (git-ignored)
```

## Roadmap
1. Round 1 (HR): landmark ROIs (MediaPipe), evaluate on UBFC-rPPG / PURE / rppg_biases with rPPG-Toolbox, address C4 (low-res) + C5 (low fps/bitrate) via simulation, report per-skin-tone (C2).
2. Round 2 (+SpO2): ratio-of-ratios on R/B AC/DC of the facial rPPG; validate on PURE / MMPD.
3. Round 3 (+BP, real-time): PTT/morphology features or learned model (vbpe / FusionVitals), Raspberry Pi demo under ≥ 2 constraints.
