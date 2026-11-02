#!/usr/bin/env python
"""Live heart-rate from the Mac webcam.

    conda run -n ML python scripts/webcam_hr.py                 # green (default), POS and CHROM selectable with --method
    conda run -n ML python scripts/webcam_hr.py --method pos --win 15
    conda run -n ML python scripts/webcam_hr.py --model models/trained/tscan_scamps300_ep5.pth   # add TS-CAN (needs torch)

Shows the camera, the ROI boxes, the last `--win` seconds of pulse signal and the HR estimate.
Keys: q quit, r reset buffer, 1/2/3 switch method (green/chrom/pos).
HR is only shown once `--win` seconds of face have been collected; it refreshes every `--step` seconds.
Sit still, face the camera, good light, 40-60 cm.
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from facevitals.roi import FaceROIExtractor  # noqa: E402
from facevitals.rppg import METHODS, bandpass, estimate_hr_fft  # noqa: E402

ROI_COLORS = {"forehead": (0, 255, 0), "left_cheek": (255, 160, 0), "right_cheek": (0, 160, 255)}


class TSCANRunner:
    """Optional learned pulse extractor (rPPG-Toolbox TS-CAN weights)."""

    def __init__(self, weights: str):
        import torch
        from facevitals.models.ts_can import TSCAN

        self.torch = torch
        self.net = TSCAN(frame_depth=10, img_size=72).eval()
        sd = torch.load(weights, map_location="cpu")
        self.net.load_state_dict({k.replace("module.", ""): v for k, v in sd.items()})

    def pulse(self, frames: np.ndarray) -> np.ndarray:
        """frames: [N,72,72,3] float32 RGB. Returns diff-normalised pulse of length N (multiple of 10)."""
        n = (len(frames) // 10) * 10
        f = frames[:n]
        d = (f[1:] - f[:-1]) / (f[1:] + f[:-1] + 1e-7)
        d = d / (d.std() + 1e-12)
        d = np.append(d, np.zeros((1,) + d.shape[1:], np.float32), 0)
        s = (f - f.mean()) / (f.std() + 1e-12)
        x = np.concatenate([d, s], -1).astype(np.float32)
        x = self.torch.from_numpy(x).permute(0, 3, 1, 2).contiguous()
        with self.torch.no_grad():
            out = self.net(x).squeeze().numpy()
        return np.cumsum(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--method", choices=list(METHODS), default="green")
    ap.add_argument("--win", type=float, default=12.0, help="analysis window (s)")
    ap.add_argument("--step", type=float, default=1.0, help="HR refresh interval (s)")
    ap.add_argument("--model", default=None, help="TS-CAN .pth to run alongside the classical method")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        raise SystemExit("cannot open camera; check macOS camera permission for the terminal app")
    ext = FaceROIExtractor()
    tscan = TSCANRunner(args.model) if args.model else None

    maxlen = int(args.win * 60)  # generous; actual fps measured
    rgb_buf: deque = deque(maxlen=maxlen)
    t_buf: deque = deque(maxlen=maxlen)
    face_buf: deque = deque(maxlen=maxlen)
    method = args.method
    hr_txt, hr_model_txt, last_est = "collecting...", "", 0.0
    pulse_disp = np.zeros(0)
    fps_est = 30.0
    t_prev = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        now = time.time()
        fps_est = 0.9 * fps_est + 0.1 / max(now - t_prev, 1e-3)
        t_prev = now
        rois = ext.rois(frame)
        if rois:
            per = ext.mean_rgb_per_roi(frame)
            rgb_buf.append(np.mean(list(per.values()), 0))
            t_buf.append(now)
            if tscan is not None:
                x0, y0, x1, y1 = ext.roi_box(frame)
                face = frame[max(0, y0 - 40) : y1 + 40, max(0, x0 - 20) : x1 + 20]
                face_buf.append(cv2.resize(cv2.cvtColor(face, cv2.COLOR_BGR2RGB), (72, 72), interpolation=cv2.INTER_AREA).astype(np.float32))
            for name, (x0, y0, x1, y1) in rois.items():
                cv2.rectangle(frame, (x0, y0), (x1, y1), ROI_COLORS.get(name, (255, 255, 255)), 2)
        else:
            cv2.putText(frame, "no face", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        # HR update every --step seconds once the window is full
        if len(t_buf) > 2 and (t_buf[-1] - t_buf[0]) >= args.win and (now - last_est) >= args.step:
            last_est = now
            ts = np.array(t_buf)
            fs = len(ts) / (ts[-1] - ts[0])
            rgb = np.array(rgb_buf)
            # resample to uniform grid (webcam timing jitters)
            tu = np.linspace(ts[0], ts[-1], int(round((ts[-1] - ts[0]) * fs)))
            rgb_u = np.stack([np.interp(tu, ts, rgb[:, c]) for c in range(3)], 1)
            pulse = METHODS[method](rgb_u, fs)
            hr = estimate_hr_fft(pulse, fs)
            hr_txt = f"{method.upper()} HR {hr:5.1f} bpm  ({fs:.1f} fps, {args.win:.0f} s)"
            pulse_disp = pulse[-int(5 * fs) :]
            if tscan is not None and len(face_buf) >= 10 * 3:
                p = tscan.pulse(np.array(face_buf))
                p = bandpass(p - p.mean(), fs)
                hr_model_txt = f"TS-CAN HR {estimate_hr_fft(p, fs):5.1f} bpm"

        # overlay
        cv2.putText(frame, hr_txt, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        if hr_model_txt:
            cv2.putText(frame, hr_model_txt, (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 255), 2)
        fill = min(1.0, (t_buf[-1] - t_buf[0]) / args.win) if len(t_buf) > 1 else 0.0
        cv2.rectangle(frame, (20, frame.shape[0] - 30), (20 + int(300 * fill), frame.shape[0] - 15), (0, 255, 0), -1)
        cv2.rectangle(frame, (20, frame.shape[0] - 30), (320, frame.shape[0] - 15), (255, 255, 255), 1)
        if len(pulse_disp) > 2:
            h, w = 100, 400
            x0, y0 = frame.shape[1] - w - 20, 20
            pd = (pulse_disp - pulse_disp.min()) / (np.ptp(pulse_disp) + 1e-9)
            pts = np.stack([x0 + np.linspace(0, w, len(pd)), y0 + h - pd * h], 1).astype(np.int32)
            cv2.rectangle(frame, (x0, y0), (x0 + w, y0 + h), (60, 60, 60), -1)
            cv2.polylines(frame, [pts], False, (0, 255, 0), 2)
        cv2.imshow("FaceVitals - press q to quit", frame)
        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break
        if k == ord("r"):
            rgb_buf.clear(); t_buf.clear(); face_buf.clear(); hr_txt, hr_model_txt = "collecting...", ""
        if k in (ord("1"), ord("2"), ord("3")):
            method = {ord("1"): "green", ord("2"): "chrom", ord("3"): "pos"}[k]
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
