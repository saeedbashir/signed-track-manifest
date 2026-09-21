"""Turn a cluster of boxes into one fixed rectangle, and classify the source layout."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt

from stm.model import Rect, SourceLayout
from stm_pipeline.config import PipelineConfig
from stm_pipeline.types import Box

Edge = Literal["left", "right"]


def _even_floor(n: int) -> int:
    return n - (n % 2)


def fixed_rect(
    boxes: Sequence[Box],
    frame_w: int,
    frame_h: int,
    low_pct: float = 5.0,
    high_pct: float = 95.0,
    pad: float = 0.08,
) -> Rect:
    """One rectangle for the whole title.

    Each edge sits at a percentile of that edge's positions across the video —
    left and top at ``low_pct``, right and bottom at ``high_pct`` — so a handful
    of outlier detections cannot inflate the crop. Then pad, clamp to the frame
    and snap to even pixel dimensions. Do not track a moving crop: the wobble
    is far more distracting than a slightly generous fixed box.
    """
    if not boxes:
        raise ValueError("cannot fit a rectangle to zero boxes")
    x1 = float(np.percentile([b.x1 for b in boxes], low_pct))
    y1 = float(np.percentile([b.y1 for b in boxes], low_pct))
    x2 = float(np.percentile([b.x2 for b in boxes], high_pct))
    y2 = float(np.percentile([b.y2 for b in boxes], high_pct))
    w = max(1.0, x2 - x1)
    h = max(1.0, y2 - y1)
    x1 -= w * pad
    x2 += w * pad
    y1 -= h * pad
    y2 += h * pad
    xi = max(0, _even_floor(int(np.floor(x1))))
    yi = max(0, _even_floor(int(np.floor(y1))))
    x2i = min(frame_w, int(np.ceil(x2)))
    y2i = min(frame_h, int(np.ceil(y2)))
    wi = max(2, _even_floor(x2i - xi))
    hi = max(2, _even_floor(y2i - yi))
    if xi + wi > frame_w:
        wi = _even_floor(frame_w - xi)
    if yi + hi > frame_h:
        hi = _even_floor(frame_h - yi)
    return Rect(xi, yi, max(2, wi), max(2, hi))


def pad_rect(rect: Rect, fraction: float, frame_w: int, frame_h: int) -> Rect:
    """Grow a rectangle outward by ``fraction`` of its own size on every side, clamped, even."""
    dx = int(np.ceil(rect.w * fraction))
    dy = int(np.ceil(rect.h * fraction))
    x = max(0, _even_floor(rect.x - dx))
    y = max(0, _even_floor(rect.y - dy))
    x2 = min(frame_w, rect.x2 + dx)
    y2 = min(frame_h, rect.y2 + dy)
    return Rect(x, y, max(2, _even_floor(x2 - x)), max(2, _even_floor(y2 - y)))


def scale_rect(rect: Rect, factor: float, frame_w: int, frame_h: int) -> Rect:
    """Scale a rectangle (e.g. sampled px -> source px), clamp and keep dimensions even."""
    x = max(0, _even_floor(int(np.floor(rect.x * factor))))
    y = max(0, _even_floor(int(np.floor(rect.y * factor))))
    x2 = min(frame_w, int(np.ceil(rect.x2 * factor)))
    y2 = min(frame_h, int(np.ceil(rect.y2 * factor)))
    return Rect(x, y, max(2, _even_floor(x2 - x)), max(2, _even_floor(y2 - y)))


def _band_stats(
    var: npt.NDArray[np.float32], mean: npt.NDArray[np.float32]
) -> tuple[float, float] | None:
    """Temporal variance and mean grey of one band, or None when it is empty."""
    if var.size == 0 or mean.size == 0:
        return None
    return float(var.mean()), float(mean.mean())


def _step(
    var: npt.NDArray[np.float32],
    mean: npt.NDArray[np.float32],
    reference: float | None,
    config: PipelineConfig,
) -> float | None:
    """Should the edge move onto this band? Returns its mean grey if so.

    A band joins the window when it is temporally static -- the backdrop does
    not move while the programme does -- and, once we have a reference, when its
    grey matches the band we accepted last.
    """
    stats = _band_stats(var, mean)
    if stats is None:
        return None
    variance, grey = stats
    if variance > config.window_var_threshold:
        return None
    if reference is not None and abs(grey - reference) > config.window_mean_delta:
        return None
    return grey


def grow_window(
    rect: Rect,
    temporal_var: npt.NDArray[np.float32],
    mean_gray: npt.NDArray[np.float32],
    frame_w: int,
    frame_h: int,
    config: PipelineConfig,
) -> Rect:
    """Expand a person rectangle outward to the region the interpreter occupies.

    An interpreter is presented in a window: a person in front of a plain
    backdrop, either composited over the programme as a rectangle (corner inset)
    or in a strip beside it (side panel). Lifting only the person's box would
    leave a floating patch of backdrop behind, so the thing to cut and to fill is
    the window, not the person.

    Each edge steps outward across bands that are static and consistent in grey,
    following the backdrop until the programme interrupts it. Growth is capped
    twice, relative to the box and absolutely as a fraction of frame width, so a
    programme that happens to be flat and static cannot pull the window across
    the screen.

    Run this before classifying the layout, not after: whether a window is flush
    against a frame edge is a property of the window, and a person standing
    inside a side panel has margin on both sides and is never flush.

    KNOWN LIMITATION, measured on a real briefing 2026-09-21. A person detector
    routinely returns a box a few pixels PAST the window border. Growth seeded
    from there references the programme rather than the backdrop and follows it
    outward -- on the Boston clip, 140 source pixels down a uniform wall. The
    border itself is unmistakable in the data (a 119-level step in mean grey
    against under 10 of variation inside the backdrop), so an edge that could
    move inward as well as outward would find it. A border-snapping
    implementation was tried and fixed that clip, but regressed four of six
    synthetic layouts, so it is not in. Settle this with the harness once there
    are real clips on both layouts; do not tune it on one.
    """
    b = max(1, config.window_band)
    x1, y1, x2, y2 = rect.x, rect.y, rect.x2, rect.y2
    max_dx = int(rect.w * config.window_max_growth)
    max_dy = int(rect.h * config.window_max_growth)
    # Absolute width backstop. Height is left to the frame, since a side panel
    # legitimately runs the full height of the picture.
    max_dx = min(max_dx, int(max(0.0, frame_w * config.window_max_frame_frac - rect.w) / 2.0))
    lx, ly = max(0, x1 - max_dx), max(0, y1 - max_dy)
    hx, hy = min(frame_w, x2 + max_dx), min(frame_h, y2 + max_dy)

    # One reference grey per edge, seeded by the first band each edge accepts.
    ref_l: float | None = None
    ref_r: float | None = None
    ref_t: float | None = None
    ref_b: float | None = None

    moved = True
    while moved:
        moved = False
        if x1 - b >= lx:
            grey = _step(
                temporal_var[y1:y2, x1 - b : x1], mean_gray[y1:y2, x1 - b : x1], ref_l, config
            )
            if grey is not None:
                x1 -= b
                ref_l = grey
                moved = True
        if x2 + b <= hx:
            grey = _step(
                temporal_var[y1:y2, x2 : x2 + b], mean_gray[y1:y2, x2 : x2 + b], ref_r, config
            )
            if grey is not None:
                x2 += b
                ref_r = grey
                moved = True
        if y1 - b >= ly:
            grey = _step(
                temporal_var[y1 - b : y1, x1:x2], mean_gray[y1 - b : y1, x1:x2], ref_t, config
            )
            if grey is not None:
                y1 -= b
                ref_t = grey
                moved = True
        if y2 + b <= hy:
            grey = _step(
                temporal_var[y2 : y2 + b, x1:x2], mean_gray[y2 : y2 + b, x1:x2], ref_b, config
            )
            if grey is not None:
                y2 += b
                ref_b = grey
                moved = True

    xi, yi = _even_floor(x1), _even_floor(y1)
    return Rect(xi, yi, max(2, _even_floor(x2 - xi)), max(2, _even_floor(y2 - yi)))


@dataclass(frozen=True)
class Layout:
    kind: SourceLayout
    edge: Edge | None = None
    #: For side panels: the full-height strip (in the same coordinates as the input rect)
    #: that holds the interpreter and should be removed from the main picture.
    panel: Rect | None = None

    def main_keep_rect(self, frame_w: int, frame_h: int) -> Rect | None:
        """For side panels, the part of the frame that remains the main programme."""
        if self.kind is not SourceLayout.SIDE_PANEL or self.panel is None:
            return None
        if self.edge == "left":
            return Rect(self.panel.x2, 0, frame_w - self.panel.x2, frame_h)
        return Rect(0, 0, self.panel.x, frame_h)


def classify_layout(
    rect: Rect,
    frame_w: int,
    frame_h: int,
    temporal_var: npt.NDArray[np.float32] | None,
    mean_gray: npt.NDArray[np.float32] | None,
    config: PipelineConfig,
) -> Layout:
    """Side panel or corner inset?

    A side panel is flush against a side edge and either spans nearly the full
    height or sits in a strip whose pixels outside the person are near-uniform
    over time (a flat backdrop). Anything else is a corner inset composited
    over the picture.
    """
    tol = round(config.edge_tolerance * frame_w)
    edge: Edge | None = None
    if rect.x <= tol:
        edge = "left"
    elif rect.x2 >= frame_w - tol:
        edge = "right"
    if edge is None:
        return Layout(SourceLayout.CORNER_INSET)

    panel = (
        Rect(0, 0, rect.x2, frame_h)
        if edge == "left"
        else Rect(rect.x, 0, frame_w - rect.x, frame_h)
    )

    if rect.h >= config.full_height_fraction * frame_h:
        return Layout(SourceLayout.SIDE_PANEL, edge, panel)

    if temporal_var is None or mean_gray is None:
        return Layout(SourceLayout.CORNER_INSET)

    # Pixels in the strip above and below the person box.
    above_v = temporal_var[0 : rect.y, panel.x : panel.x2]
    below_v = temporal_var[rect.y2 : frame_h, panel.x : panel.x2]
    above_m = mean_gray[0 : rect.y, panel.x : panel.x2]
    below_m = mean_gray[rect.y2 : frame_h, panel.x : panel.x2]
    outside_rows = above_v.shape[0] + below_v.shape[0]
    if outside_rows < 0.05 * frame_h:
        return Layout(SourceLayout.CORNER_INSET)
    var_mean = float(np.concatenate([above_v.ravel(), below_v.ravel()]).mean())
    spatial_std = float(np.concatenate([above_m.ravel(), below_m.ravel()]).std())
    if var_mean <= config.panel_var_threshold and spatial_std <= config.panel_std_threshold:
        return Layout(SourceLayout.SIDE_PANEL, edge, panel)
    return Layout(SourceLayout.CORNER_INSET)
