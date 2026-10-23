"""Face detection and ROI mean-color extraction.

Uses OpenCV's bundled Haar cascade so there are no extra dependencies. The ROI is the
upper-face box (forehead + cheeks region) which is what the task spec asks for. Swap in
MediaPipe FaceMesh later for landmark-based forehead / cheek polygons (Task 1 spec) and
for C7 mask handling.
"""
from __future__ import annotations

import cv2
import numpy as np


class FaceROIExtractor:
    def __init__(self, smoothing: float = 0.7, redetect_every: int = 30):
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.detector = cv2.CascadeClassifier(cascade_path)
        self.smoothing = smoothing
        self.redetect_every = redetect_every
        self._box: np.ndarray | None = None
        self._frame_idx = 0

    def _detect(self, gray: np.ndarray) -> np.ndarray | None:
        faces = self.detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
        if len(faces) == 0:
            return None
        # largest face = designated subject (C3 will replace this with tracking)
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        return np.array([x, y, w, h], dtype=float)

    def roi_box(self, frame_bgr: np.ndarray) -> tuple[int, int, int, int] | None:
        """Return (x0, y0, x1, y1) of the skin ROI for this frame, or None if no face."""
        if self._box is None or self._frame_idx % self.redetect_every == 0:
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            det = self._detect(gray)
            if det is not None:
                self._box = det if self._box is None else self.smoothing * self._box + (1 - self.smoothing) * det
        self._frame_idx += 1
        if self._box is None:
            return None
        x, y, w, h = self._box
        # central 60% width, top 15%..65% height -> forehead + cheeks, avoids eyes/mouth edges
        x0, x1 = int(x + 0.2 * w), int(x + 0.8 * w)
        y0, y1 = int(y + 0.15 * h), int(y + 0.65 * h)
        return x0, y0, x1, y1

    def mean_rgb(self, frame_bgr: np.ndarray) -> np.ndarray | None:
        box = self.roi_box(frame_bgr)
        if box is None:
            return None
        x0, y0, x1, y1 = box
        patch = frame_bgr[y0:y1, x0:x1]
        if patch.size == 0:
            return None
        b, g, r = patch.reshape(-1, 3).mean(axis=0)
        return np.array([r, g, b], dtype=float)


def extract_rgb_trace(video_path: str, max_frames: int | None = None):
    """Read a video and return (traces [N,3] RGB, fps, boxes list)."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    ext = FaceROIExtractor()
    traces, boxes = [], []
    last = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        rgb = ext.mean_rgb(frame)
        if rgb is None:
            rgb = last if last is not None else np.array([np.nan] * 3)
        last = rgb
        traces.append(rgb)
        boxes.append(ext.roi_box(frame))
        if max_frames and len(traces) >= max_frames:
            break
    cap.release()
    traces = np.array(traces)
    # forward/backward-fill NaNs from frames before first detection
    if np.isnan(traces).any():
        for c in range(3):
            col = traces[:, c]
            idx = np.where(~np.isnan(col))[0]
            if len(idx):
                col[: idx[0]] = col[idx[0]]
    return traces, fps, boxes
