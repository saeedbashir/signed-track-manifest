"""Draw a ground-truth rectangle on one frame with the mouse and save it."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt

from stm.model import Rect
from stm_pipeline.harness.ground_truth import GroundTruth, load_ground_truth, save_ground_truth


def grab_frame(clip: Path, at_s: float) -> npt.NDArray[np.uint8]:
    cap = cv2.VideoCapture(str(clip))
    try:
        cap.set(cv2.CAP_PROP_POS_MSEC, at_s * 1000.0)
        ok, frame = cap.read()
        if not ok or frame is None:
            raise OSError(f"could not read a frame from {clip} at {at_s}s")
        image: npt.NDArray[np.uint8] = np.asarray(frame, dtype=np.uint8)
        return image
    finally:
        cap.release()


def select_rect(frame: npt.NDArray[np.uint8], title: str) -> Rect | None:
    """Open a window, let the user drag a rectangle, press Enter. Returns None on Escape."""
    x, y, w, h = cv2.selectROI(title, frame, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow(title)
    if w <= 0 or h <= 0:
        return None
    return Rect(int(x), int(y), int(w), int(h))


def label(
    clip: Path, gt_path: Path, at_s: float = 5.0, with_face: bool = False
) -> GroundTruth | None:
    frame = grab_frame(clip, at_s)
    rect = select_rect(frame, "Drag the interpreter box, press Enter")
    if rect is None:
        return None
    face = None
    if with_face:
        face = select_rect(frame, "Drag the face box at its most expressive, press Enter")
    items = [g for g in (load_ground_truth(gt_path) if gt_path.exists() else []) if g.clip != clip]
    gt = GroundTruth(clip=clip, rect=rect, face=face)
    items.append(gt)
    save_ground_truth(gt_path, items)
    return gt
