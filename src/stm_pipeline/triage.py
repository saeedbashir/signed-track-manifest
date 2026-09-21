"""Rank candidate titles by how good a demo they will make.

Experiment 1 asks whether a usable pool of interpreted content exists. Whether
we *may* use a title is a licence question and stays with a person. Whether a
title will *look good* is measurable, and this measures it.

The implementation spec ranks selection criteria, and the first one is "corner
inset with a clean, low-detail region behind the interpreter -- the easier the
fill, the better the result". Fill quality is the highest-likelihood,
highest-impact risk in the project: the vacated rectangle is patched by
interpolating from its surroundings, and interpolation into busy, moving
pixels smears. So the number that matters is how busy the ring around the
interpreter's window is, in space and in time.

This runs detection only -- no encoding -- so a candidate is judged in about a
second rather than a full pipeline run.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt

from stm.model import LOW_CONFIDENCE_THRESHOLD, Rect, SourceLayout
from stm_pipeline.config import PipelineConfig
from stm_pipeline.crop import scale_rect
from stm_pipeline.detect.base import Detector
from stm_pipeline.ffmpeg import FfmpegError
from stm_pipeline.run import Decision, NoSignerFoundError, decide

#: Width of the ring sampled around the window, as a fraction of the window.
RING_FRACTION = 0.25
#: Above this, interpolation has busy pixels to copy from and the patch shows.
DETAIL_EASY = 12.0
DETAIL_HARD = 36.0
#: Temporal variance in the ring. A moving background makes the patch flicker.
MOTION_EASY = 8.0
MOTION_HARD = 60.0


def _ring_mask(rect: Rect, w: int, h: int) -> tuple[slice, slice, Rect]:
    """The bounding box of the ring around ``rect``, clipped to the frame."""
    dx = max(2, int(rect.w * RING_FRACTION))
    dy = max(2, int(rect.h * RING_FRACTION))
    x1, y1 = max(0, rect.x - dx), max(0, rect.y - dy)
    x2, y2 = min(w, rect.x2 + dx), min(h, rect.y2 + dy)
    return slice(y1, y2), slice(x1, x2), Rect(x1, y1, max(1, x2 - x1), max(1, y2 - y1))


def _normalise(value: float, easy: float, hard: float) -> float:
    """0 where the fill is easy, 1 where it is hard."""
    if hard <= easy:
        return 0.0
    return float(min(1.0, max(0.0, (value - easy) / (hard - easy))))


@dataclass(frozen=True)
class Triage:
    clip: str
    ok: bool
    layout: str | None = None
    edge: str | None = None
    rect: Rect | None = None
    width: int = 0
    height: int = 0
    duration_s: float = 0.0
    confidence: float = 0.0
    size_frac: float = 0.0
    detail: float = 0.0
    motion: float = 0.0
    fill_difficulty: float = 1.0
    error: str | None = None

    @property
    def verdict(self) -> str:
        """A demo-quality call, not a licence call."""
        if not self.ok:
            return "unusable"
        if self.layout == str(SourceLayout.SIDE_PANEL):
            # Removing a strip needs no reconstruction at all.
            return "excellent"
        if self.confidence < LOW_CONFIDENCE_THRESHOLD:
            return "marginal"
        if self.fill_difficulty < 0.35:
            return "good"
        if self.fill_difficulty < 0.65:
            return "marginal"
        return "poor"

    @property
    def rank(self) -> tuple[int, float]:
        order = {"excellent": 0, "good": 1, "marginal": 2, "poor": 3, "unusable": 4}
        return order[self.verdict], self.fill_difficulty

    def to_dict(self) -> dict[str, Any]:
        return {
            "clip": self.clip,
            "verdict": self.verdict,
            "layout": self.layout,
            "edge": self.edge,
            "rect": self.rect.to_dict() if self.rect else None,
            "resolution": f"{self.width}x{self.height}" if self.width else None,
            "durationS": round(self.duration_s, 1),
            "confidence": round(self.confidence, 3),
            "sizeFrac": round(self.size_frac, 3),
            "backgroundDetail": round(self.detail, 1),
            "backgroundMotion": round(self.motion, 1),
            "fillDifficulty": round(self.fill_difficulty, 3),
            "error": self.error,
        }


def measure_background(decision: Decision, rect_sample: Rect) -> tuple[float, float, float]:
    """Spatial detail and temporal variance of the ring around the window.

    Both matter and they fail differently. Spatial detail decides whether the
    patch looks like a smear in a single frame; temporal variance decides
    whether it flickers between frames. A busy still wall is recoverable; moving
    footage behind the inset is not.
    """
    a = decision.analysis
    if a is None:
        return 0.0, 0.0, 0.5
    rows, cols, ring = _ring_mask(rect_sample, a.sample_width, a.sample_height)
    mean: npt.NDArray[np.float32] = a.mean_gray[rows, cols]
    var: npt.NDArray[np.float32] = a.temporal_var[rows, cols]
    if mean.size == 0:
        return 0.0, 0.0, 0.5

    # Exclude the window itself: only the surroundings are copied from.
    keep = np.ones(mean.shape, dtype=bool)
    iy1 = max(0, rect_sample.y - ring.y)
    ix1 = max(0, rect_sample.x - ring.x)
    keep[iy1 : iy1 + rect_sample.h, ix1 : ix1 + rect_sample.w] = False
    if not keep.any():
        return 0.0, 0.0, 0.5

    gx = cv2.Sobel(mean, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(mean, cv2.CV_32F, 0, 1, ksize=3)
    detail = float(np.sqrt(gx * gx + gy * gy)[keep].mean())
    motion = float(var[keep].mean())
    difficulty = 0.6 * _normalise(detail, DETAIL_EASY, DETAIL_HARD) + 0.4 * _normalise(
        motion, MOTION_EASY, MOTION_HARD
    )
    return detail, motion, difficulty


def triage(clip: Path, detector: Detector, config: PipelineConfig) -> Triage:
    try:
        d = decide(clip, detector, config)
    except (NoSignerFoundError, FfmpegError, OSError, ValueError) as exc:
        # One unreadable or unusable candidate must not end the batch: triage is
        # run over a directory of downloads, and a bad file is a result, not a crash.
        return Triage(clip=str(clip), ok=False, error=f"{type(exc).__name__}: {exc}")

    a = d.analysis
    if a is None:
        return Triage(clip=str(clip), ok=False, error="no analysis (manual crop?)")

    scale = a.sample_scale or 1.0
    rect_sample = scale_rect(d.rect_source, scale, a.sample_width, a.sample_height)
    detail, motion, difficulty = measure_background(d, rect_sample)
    if d.layout.kind is SourceLayout.SIDE_PANEL:
        # A strip is removed, not patched, so there is nothing to interpolate.
        difficulty = 0.0

    return Triage(
        clip=str(clip),
        ok=True,
        layout=str(d.layout.kind),
        edge=d.layout.edge,
        rect=d.rect_source,
        width=a.info.width,
        height=a.info.height,
        duration_s=a.info.duration_s,
        confidence=d.confidence,
        size_frac=d.rect_source.w / a.info.width if a.info.width else 0.0,
        detail=detail,
        motion=motion,
        fill_difficulty=difficulty,
    )


def format_table(results: list[Triage]) -> str:
    rows = sorted(results, key=lambda r: r.rank)
    lines = [
        f"{'clip':30} {'verdict':10} {'layout':13} {'fill':>5} {'conf':>5} "
        f"{'size':>5} {'detail':>7} {'motion':>7} {'dur':>6}"
    ]
    for r in rows:
        if not r.ok:
            lines.append(f"{Path(r.clip).name[:30]:30} {'unusable':10} {r.error or ''}")
            continue
        lines.append(
            f"{Path(r.clip).name[:30]:30} {r.verdict:10} "
            f"{(r.layout or '') + (f'/{r.edge}' if r.edge else ''):13} "
            f"{r.fill_difficulty:5.2f} {r.confidence:5.2f} {r.size_frac:5.2f} "
            f"{r.detail:7.1f} {r.motion:7.1f} {r.duration_s:6.0f}s"
        )
    good = sum(1 for r in results if r.verdict in ("excellent", "good"))
    lines.append(
        f"\n{good} of {len(results)} would make a clean demo "
        f"(fill difficulty below 0.35, or a side panel that needs no fill at all)."
    )
    lines.append(
        "Licence, production credit and how the file was obtained are not measured here. "
        "They stay in sources.yaml, and they are your call."
    )
    return "\n".join(lines)


def suggest_sources_entry(r: Triage) -> str:
    """A sources.yaml skeleton with the observations filled in and the judgement left blank."""
    stem = Path(r.clip).stem.lower().replace(" ", "-").replace("_", "-")[:60]
    layout_note = f"{r.layout}" + (f", {r.edge} edge" if r.edge else "")
    ease = (
        "clean, low detail"
        if r.fill_difficulty < 0.35
        else ("moderate detail" if r.fill_difficulty < 0.65 else "busy: fill will show")
    )
    return (
        f"  - id: {stem}\n"
        f'    title: ""                      # TODO\n'
        f'    url: ""                        # TODO where it was published\n'
        f'    publisher: ""                  # TODO\n'
        f'    production_credit: ""          # TODO in-house or a contractor?\n'
        f'    licence: ""                    # TODO your call\n'
        f'    attribution: ""                # TODO shown on the title detail screen\n'
        f'    acquisition_route: ""          # TODO agency download / archive / request\n'
        f'    layout_notes: "{layout_note}; behind the inset: {ease}"\n'
        f'    framing_notes: ""              # TODO waist-up with the face fully visible?\n'
        f"    duration_s: {r.duration_s:.0f}\n"
        f"    verdict: candidate\n"
        f'    notes: "triage fill {r.fill_difficulty:.2f}, confidence {r.confidence:.2f}"\n'
    )
