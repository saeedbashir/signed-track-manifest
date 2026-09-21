"""Single streaming pass over a video: detect people, measure motion, accumulate pixel stats.

Nothing frame-sized is retained except the previous grey frame and two running
statistics images, so a 20-minute 1080p source costs a few megabytes of memory.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt

from stm_pipeline.config import PipelineConfig
from stm_pipeline.detect.base import Detector
from stm_pipeline.ffmpeg import probe
from stm_pipeline.sample import iter_sampled_frames
from stm_pipeline.types import Box, FrameDetections, FrameInfo


@dataclass
class Analysis:
    info: FrameInfo
    detector_name: str
    sample_every: int
    sample_scale: float
    sample_width: int
    sample_height: int
    frames: list[FrameDetections]
    mean_gray: npt.NDArray[np.float32]
    temporal_var: npt.NDArray[np.float32]

    @property
    def sampled_count(self) -> int:
        return len(self.frames)


def box_motion(diff: npt.NDArray[np.float32], box: Box, upper_fraction: float = 0.6) -> float:
    """Mean absolute frame difference inside the upper part of ``box``.

    Hands and face live in the upper part of a person box; the lower part is
    torso and background. NaN when the region is empty.
    """
    h, w = diff.shape[:2]
    x1 = max(0, math.floor(box.x1))
    y1 = max(0, math.floor(box.y1))
    x2 = min(w, math.ceil(box.x2))
    y2 = min(h, math.ceil(box.y1 + box.h * upper_fraction))
    if x2 <= x1 or y2 <= y1:
        return math.nan
    return float(diff[y1:y2, x1:x2].mean())


def analyze(path: Path, detector: Detector, config: PipelineConfig) -> Analysis:
    info = probe(path)
    frames: list[FrameDetections] = []
    prev_gray: npt.NDArray[np.float32] | None = None
    mean: npt.NDArray[np.float32] | None = None
    m2: npt.NDArray[np.float32] | None = None
    n = 0
    scale = 1.0
    sample_w = info.width
    sample_h = info.height

    for sf in iter_sampled_frames(path, config.sample_every, config.sample_width):
        scale = sf.scale
        sample_w, sample_h = sf.width, sf.height
        boxes = tuple(detector.detect(sf))
        gray = cv2.cvtColor(sf.image, cv2.COLOR_BGR2GRAY).astype(np.float32)

        if prev_gray is None or prev_gray.shape != gray.shape:
            motions = tuple(math.nan for _ in boxes)
            frame_motion = math.nan
        else:
            diff = np.abs(gray - prev_gray)
            frame_motion = float(diff.mean())
            motions = tuple(box_motion(diff, b, config.motion_upper_fraction) for b in boxes)
        prev_gray = gray

        # Welford running mean and variance per pixel.
        n += 1
        if mean is None or m2 is None:
            mean = gray.copy()
            m2 = np.zeros_like(gray)
        else:
            delta = gray - mean
            mean += delta / n
            m2 += delta * (gray - mean)

        frames.append(
            FrameDetections(
                index=sf.index,
                time_s=sf.time_s,
                boxes=boxes,
                motions=motions,
                frame_motion=frame_motion,
            )
        )

    if mean is None or m2 is None:
        raise OSError(f"no frames could be read from {path}")
    var = m2 / max(1, n - 1)
    return Analysis(
        info=info,
        detector_name=detector.name,
        sample_every=config.sample_every,
        sample_scale=scale,
        sample_width=sample_w,
        sample_height=sample_h,
        frames=frames,
        mean_gray=mean,
        temporal_var=var.astype(np.float32),
    )
