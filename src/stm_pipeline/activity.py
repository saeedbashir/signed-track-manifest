"""Signing activity: how much the interpreter's hands and face move, per second.

Published in the manifest so a player can do the one thing the pipeline cannot:
react while the programme plays. The first use is an opt-in fade of the signer
layer while the interpreter is idle. The number describes hand motion and
nothing else — not meaning, not quality, not whether the interpretation is any
good — and the format says so.

Measured from the *final* crop rectangle rather than from the detector's
clusters. The titles that actually ship are hand-measured (``manual-crop``) and
never pass through detection at all; a signal that existed only for detected
titles would exist only for the titles a viewer never sees. One streaming pass
over the crop, the same for both methods, and ``stm activity`` can rerun it on
its own.

Normalised per title to its own 95th percentile. Mean absolute grey difference
is in units that depend on resolution, compression and lighting, so no fixed
scale means the same thing across two broadcasters. Within one title, 100 is
"as active as this interpreter gets" and 0 is still. The threshold for *idle*
belongs to the player, not here — the same principle as ``confidence``: publish
the measurement, let consumers set the rule.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from stm.model import Activity, Rect
from stm_pipeline.analyze import box_motion
from stm_pipeline.config import PipelineConfig
from stm_pipeline.ffmpeg import probe
from stm_pipeline.sample import iter_sampled_frames
from stm_pipeline.types import Box

#: Nine levels, so a track can be eyeballed in a terminal.
_BARS = " ▁▂▃▄▅▆▇█"


@dataclass(frozen=True)
class ActivityMeasurement:
    """An :class:`Activity` plus what it took to make it, for the analysis report."""

    activity: Activity
    #: The raw motion value that became 100. Recorded so two titles can be compared.
    p95: float
    #: Sampled frames that contributed a motion value.
    samples: int

    def to_dict(self) -> dict[str, object]:
        return {
            "intervalMs": self.activity.interval_ms,
            "scale": self.activity.scale,
            "length": len(self.activity.values),
            "p95": round(self.p95, 3),
            "samples": self.samples,
            "idleFraction": round(idle_fraction(self.activity.values), 3),
        }


def bucket_values(
    samples: Sequence[tuple[float, float]], interval_s: float, duration_s: float
) -> list[float]:
    """Mean motion per interval from ``(time_s, motion)`` samples, from t=0.

    Intervals with no sample — a dropped frame, or a sampling stride longer than
    the interval — repeat the previous interval's value rather than reading as
    still. Zero would be a claim about the interpreter that nothing measured.
    """
    if interval_s <= 0:
        raise ValueError("interval_s must be positive")
    count = max(1, math.ceil(max(duration_s, 0.0) / interval_s))
    sums = [0.0] * count
    counts = [0] * count
    for time_s, motion in samples:
        if math.isnan(motion) or time_s < 0:
            continue
        i = min(count - 1, int(time_s // interval_s))
        sums[i] += motion
        counts[i] += 1
    out: list[float] = []
    prev = 0.0
    for s, n in zip(sums, counts, strict=True):
        prev = s / n if n else prev
        out.append(prev)
    return out


def normalise(values: Sequence[float], percentile: float = 95.0) -> tuple[list[int], float]:
    """Values as integers 0..100 against the given percentile of themselves.

    Returns the integers and the raw value that became 100. A track with no
    motion at all normalises to zeros rather than dividing by nothing.
    """
    if not values:
        return [], 0.0
    arr = np.asarray(values, dtype=np.float64)
    top = float(np.percentile(arr, percentile))
    if not math.isfinite(top) or top <= 1e-9:
        return [0] * len(values), 0.0
    scaled = np.clip(arr / top, 0.0, 1.0) * 100.0
    return [round(float(v)) for v in scaled], top


def idle_fraction(values: Sequence[int], threshold: int = 20) -> float:
    """Share of intervals under ``threshold`` — a first look at how much idle time a title has."""
    if not values:
        return 0.0
    return sum(1 for v in values if v < threshold) / len(values)


def sparkline(values: Sequence[int], width: int = 80) -> str:
    """The track as one line of block characters, downsampled to ``width`` columns."""
    if not values:
        return ""
    if len(values) <= width:
        cols = [float(v) for v in values]
    else:
        edges = np.linspace(0, len(values), width + 1)
        cols = [
            float(np.mean(values[int(edges[i]) : max(int(edges[i]) + 1, int(edges[i + 1]))]))
            for i in range(width)
        ]
    return "".join(_BARS[min(len(_BARS) - 1, round(c / 100.0 * (len(_BARS) - 1)))] for c in cols)


def measure_activity(
    video: Path,
    rect_source: Rect,
    config: PipelineConfig,
    interval_ms: int | None = None,
) -> ActivityMeasurement:
    """Stream the video once and measure motion inside the upper part of ``rect_source``.

    The rectangle is in source pixels; frames are sampled and downscaled exactly
    as the analysis pass does, so this costs what ``analyze`` costs and no more.
    """
    interval_ms = interval_ms or config.activity_interval_ms
    interval_s = interval_ms / 1000.0
    info = probe(video)
    samples: list[tuple[float, float]] = []
    prev_gray: np.ndarray | None = None
    for sf in iter_sampled_frames(video, config.sample_every, config.sample_width):
        gray = cv2.cvtColor(sf.image, cv2.COLOR_BGR2GRAY).astype(np.float32)
        if prev_gray is not None and prev_gray.shape == gray.shape:
            s = sf.scale
            box = Box(rect_source.x * s, rect_source.y * s, rect_source.x2 * s, rect_source.y2 * s)
            motion = box_motion(np.abs(gray - prev_gray), box, config.motion_upper_fraction)
            samples.append((sf.time_s, motion))
        prev_gray = gray
    duration_s = info.duration_s if info.duration_s > 0 else (samples[-1][0] if samples else 0.0)
    means = bucket_values(samples, interval_s, duration_s)
    values, p95 = normalise(means, config.activity_percentile)
    activity = Activity(values=values, interval_ms=interval_ms)
    return ActivityMeasurement(
        activity=activity, p95=p95, samples=sum(1 for _, m in samples if not math.isnan(m))
    )
