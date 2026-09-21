"""Shared value types for the pipeline."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from stm.model import Rect


@dataclass(frozen=True)
class Box:
    """A detection box in float pixel coordinates (x1, y1, x2, y2) with a score."""

    x1: float
    y1: float
    x2: float
    y2: float
    score: float = 1.0

    @property
    def w(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def h(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2.0

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2.0

    @property
    def area(self) -> float:
        return self.w * self.h

    def iou(self, other: Box) -> float:
        ix = max(0.0, min(self.x2, other.x2) - max(self.x1, other.x1))
        iy = max(0.0, min(self.y2, other.y2) - max(self.y1, other.y1))
        inter = ix * iy
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0

    def scaled(self, factor: float) -> Box:
        return Box(
            self.x1 * factor, self.y1 * factor, self.x2 * factor, self.y2 * factor, self.score
        )

    def to_rect(self) -> Rect:
        x = round(self.x1)
        y = round(self.y1)
        return Rect(x, y, max(1, round(self.x2) - x), max(1, round(self.y2) - y))


@dataclass(frozen=True)
class FrameInfo:
    """What ffprobe tells us about the source video's first video stream."""

    width: int
    height: int
    fps: float
    duration_s: float
    codec: str
    frame_count: int | None = None

    @property
    def duration_ms(self) -> int:
        return round(self.duration_s * 1000)


@dataclass(frozen=True)
class SampledFrame:
    """One sampled frame, downscaled. ``scale`` converts sampled px to source px by division."""

    index: int
    time_s: float
    image: npt.NDArray[np.uint8]
    scale: float

    @property
    def width(self) -> int:
        return int(self.image.shape[1])

    @property
    def height(self) -> int:
        return int(self.image.shape[0])


@dataclass(frozen=True)
class FrameDetections:
    """Detections for one sampled frame, in sampled-frame coordinates.

    ``motions`` is the per-box motion energy in the upper part of the box and
    ``frame_motion`` the mean motion over the whole frame, both NaN on the
    first sampled frame where no previous frame exists.
    """

    index: int
    time_s: float
    boxes: tuple[Box, ...]
    motions: tuple[float, ...]
    frame_motion: float
