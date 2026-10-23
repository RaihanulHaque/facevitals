"""Face detection and ROI mean-color extraction.

Detector: OpenCV YuNet (`cv2.FaceDetectorYN`, ONNX in `models/`), available on OpenCV >= 4.5.4 and
required on OpenCV 5 where the Haar `CascadeClassifier` was removed. Falls back to Haar on old builds.
YuNet also returns 5 landmarks (eyes, nose tip, mouth corners), which we use to place the Task-1 ROIs:
forehead (above the eye line) and both cheeks (beside the nose, below the eyes).
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

YUNET_PATH = Path(__file__).resolve().parents[2] / "models" / "face_detection_yunet_2023mar.onnx"


class FaceROIExtractor:
    def __init__(self, smoothing: float = 0.7, redetect_every: int = 5, score_thresh: float = 0.7):
        self.smoothing = smoothing
        self.redetect_every = redetect_every
        self._det = None
        self._size = None
        self._haar = None
        if hasattr(cv2, "FaceDetectorYN") and YUNET_PATH.exists():
            self._det = cv2.FaceDetectorYN.create(str(YUNET_PATH), "", (320, 320), score_thresh, 0.3, 5000)
        elif hasattr(cv2, "CascadeClassifier"):
            self._haar = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        else:
            raise RuntimeError(f"No face detector: put YuNet ONNX at {YUNET_PATH}")
        self._face: np.ndarray | None = None  # [x, y, w, h, 5 landmarks (x,y)...] -> 14 floats
        self._frame_idx = 0

    # ------------------------------------------------------------------ detection
    def _detect(self, frame_bgr: np.ndarray) -> np.ndarray | None:
        h, w = frame_bgr.shape[:2]
        if self._det is not None:
            if self._size != (w, h):
                self._det.setInputSize((w, h))
                self._size = (w, h)
            _, faces = self._det.detect(frame_bgr)
            if faces is None or len(faces) == 0:
                return None
            f = max(faces, key=lambda r: r[2] * r[3])  # largest = designated subject (C3: replace with tracker)
            return f[:14].astype(float)
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = self._haar.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
        if len(faces) == 0:
            return None
        x, y, bw, bh = max(faces, key=lambda r: r[2] * r[3])
        # synthesize landmarks from the box so the ROI logic is shared
        lm = [x + 0.3 * bw, y + 0.4 * bh, x + 0.7 * bw, y + 0.4 * bh, x + 0.5 * bw, y + 0.6 * bh,
              x + 0.35 * bw, y + 0.8 * bh, x + 0.65 * bw, y + 0.8 * bh]
        return np.array([x, y, bw, bh, *lm], dtype=float)

    def update(self, frame_bgr: np.ndarray) -> bool:
        if self._face is None or self._frame_idx % self.redetect_every == 0:
            det = self._detect(frame_bgr)
            if det is not None:
                self._face = det if self._face is None else self.smoothing * self._face + (1 - self.smoothing) * det
        self._frame_idx += 1
        return self._face is not None

    # ------------------------------------------------------------------ ROIs
    def rois(self, frame_bgr: np.ndarray) -> dict[str, tuple[int, int, int, int]] | None:
        """Return {name: (x0, y0, x1, y1)} for forehead / left_cheek / right_cheek, or None."""
        if not self.update(frame_bgr):
            return None
        H, W = frame_bgr.shape[:2]
        x, y, w, h, rex, rey, lex, ley, nx, ny, rmx, rmy, lmx, lmy = self._face
        eye_y = 0.5 * (rey + ley)
        eye_dx = abs(lex - rex)

        def clip(x0, y0, x1, y1):
            x0, x1 = int(max(0, min(x0, x1))), int(min(W, max(x0, x1)))
            y0, y1 = int(max(0, min(y0, y1))), int(min(H, max(y0, y1)))
            return (x0, y0, x1, y1) if x1 > x0 + 2 and y1 > y0 + 2 else None

        out = {
            "forehead": clip(x + 0.25 * w, y + 0.05 * h, x + 0.75 * w, eye_y - 0.6 * eye_dx * 0.5),
            "right_cheek": clip(rex - 0.5 * eye_dx, eye_y + 0.25 * eye_dx, nx - 0.15 * eye_dx, ny + 0.35 * eye_dx),
            "left_cheek": clip(nx + 0.15 * eye_dx, eye_y + 0.25 * eye_dx, lex + 0.5 * eye_dx, ny + 0.35 * eye_dx),
        }
        return {k: v for k, v in out.items() if v is not None} or None

    def roi_box(self, frame_bgr: np.ndarray) -> tuple[int, int, int, int] | None:
        """Backwards-compatible single box: union of the ROIs."""
        r = self.rois(frame_bgr)
        if not r:
            return None
        xs0, ys0, xs1, ys1 = zip(*r.values())
        return min(xs0), min(ys0), max(xs1), max(ys1)

    def mean_rgb(self, frame_bgr: np.ndarray) -> np.ndarray | None:
        """Area-weighted mean RGB over all ROIs (R, G, B order)."""
        per = self.mean_rgb_per_roi(frame_bgr)
        if not per:
            return None
        rgb = np.array(list(per.values()))
        return rgb.mean(axis=0)

    def mean_rgb_per_roi(self, frame_bgr: np.ndarray) -> dict[str, np.ndarray] | None:
        r = self.rois(frame_bgr)
        if not r:
            return None
        out = {}
        for name, (x0, y0, x1, y1) in r.items():
            patch = frame_bgr[y0:y1, x0:x1]
            if patch.size:
                b, g, rr = patch.reshape(-1, 3).mean(axis=0)
                out[name] = np.array([rr, g, b], dtype=float)
        return out or None


def extract_rgb_trace(video_path: str, max_frames: int | None = None):
    """Read a video and return (traces [N,3] RGB, fps, per_roi {name: [N,3]})."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    ext = FaceROIExtractor()
    per_roi: dict[str, list] = {"forehead": [], "left_cheek": [], "right_cheek": []}
    last: dict[str, np.ndarray] = {}
    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        cur = ext.mean_rgb_per_roi(frame) or {}
        for name in per_roi:
            v = cur.get(name, last.get(name, np.array([np.nan] * 3)))
            per_roi[name].append(v)
            if name in cur:
                last[name] = cur[name]
        n += 1
        if max_frames and n >= max_frames:
            break
    cap.release()
    arrays = {}
    for name, rows in per_roi.items():
        a = np.array(rows, dtype=float)
        for c in range(3):  # back-fill frames before first detection
            idx = np.where(~np.isnan(a[:, c]))[0]
            if len(idx):
                a[: idx[0], c] = a[idx[0], c]
        if not np.isnan(a).all():
            arrays[name] = a
    if not arrays:
        raise RuntimeError("no face detected in any frame")
    traces = np.mean(np.stack(list(arrays.values())), axis=0)
    return traces, fps, arrays
