from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt
import pytest

from stm.model import Rect, SourceLayout
from stm_pipeline.analyze import box_motion
from stm_pipeline.confidence import confidence
from stm_pipeline.config import PipelineConfig
from stm_pipeline.crop import classify_layout, fixed_rect, grow_window, pad_rect, scale_rect
from stm_pipeline.identify import choose_signer, cluster_detections, rank_clusters
from stm_pipeline.types import Box, FrameDetections

W, H = 640, 360
CFG = PipelineConfig()


def _frames(n: int = 40) -> list[FrameDetections]:
    """An anchor filling the middle with low motion, a small stable signer with high
    motion, and a person who wanders through for a few frames."""
    frames: list[FrameDetections] = []
    for i in range(n):
        jitter = (i % 3) - 1
        signer = Box(480 + jitter, 220 + jitter, 600 + jitter, 350 + jitter, 0.9)
        anchor = Box(150, 40, 450, 350, 0.95)
        boxes = [anchor, signer]
        motions = [1.5, 12.0]
        if 10 <= i < 16:
            boxes.append(Box(20 + i * 8, 100, 120 + i * 8, 340, 0.8))
            motions.append(6.0)
        fm = math.nan if i == 0 else 3.0
        frames.append(FrameDetections(i, i / 5.0, tuple(boxes), tuple(motions), fm))
    return frames


def test_clusters_track_by_overlap() -> None:
    clusters = cluster_detections(_frames(), CFG)
    sizes = sorted(len(c.boxes) for c in clusters)
    # anchor and signer are present on all 40 frames; the walker is fragmented or short
    assert sizes[-2:] == [40, 40]


def test_signer_is_the_small_stable_high_motion_cluster() -> None:
    frames = _frames()
    clusters = cluster_detections(frames, CFG)
    ranked = rank_clusters(clusters, len(frames), W, H, CFG)
    chosen = choose_signer(ranked)
    assert chosen is not None
    assert chosen.eligible
    assert chosen.size_frac < 0.25
    assert chosen.motion_ratio > 3
    anchor = next(f for f in ranked if f.size_frac > 0.4)
    assert not anchor.eligible
    assert "size" in anchor.reason


def test_nothing_eligible_returns_none() -> None:
    frames = [FrameDetections(i, i / 5, (Box(0, 0, 600, 300),), (1.0,), 1.0) for i in range(10)]
    ranked = rank_clusters(cluster_detections(frames, CFG), 10, W, H, CFG)
    assert choose_signer(ranked) is None


def test_confidence_is_bounded_and_monotone() -> None:
    frames = _frames()
    ranked = rank_clusters(cluster_detections(frames, CFG), len(frames), W, H, CFG)
    chosen = choose_signer(ranked)
    assert chosen is not None
    c = confidence(chosen)
    assert 0.0 <= c <= 1.0
    assert c > 0.7


def test_fixed_rect_percentiles_pad_and_even() -> None:
    boxes = [Box(100, 100, 200, 220) for _ in range(19)] + [Box(0, 0, 640, 360)]  # one outlier
    r = fixed_rect(boxes, W, H, 5.0, 95.0, 0.08)
    assert r.w % 2 == 0 and r.h % 2 == 0
    assert r.x % 2 == 0 and r.y % 2 == 0
    # padded around the stable box, not blown out by the single outlier
    assert 80 <= r.x <= 94
    assert 200 <= r.x2 <= 240
    assert r.x2 <= W and r.y2 <= H


def test_fixed_rect_clamps_to_frame() -> None:
    r = fixed_rect([Box(600, 300, 640, 360)], W, H)
    assert r.x2 <= W and r.y2 <= H


def test_fixed_rect_rejects_empty() -> None:
    with pytest.raises(ValueError):
        fixed_rect([], W, H)


def test_scale_rect_keeps_even_and_in_frame() -> None:
    r = scale_rect(Rect(100, 50, 101, 51), 3.0, 1920, 1080)
    assert r.w % 2 == 0 and r.h % 2 == 0
    assert r.x2 <= 1920 and r.y2 <= 1080
    assert r.x == 300 and r.y == 150


def test_pad_rect_grows_and_clamps() -> None:
    r = pad_rect(Rect(100, 100, 100, 50), 0.1, W, H)
    assert r.x == 90 and r.y == 94  # 5 px -> even floor
    assert r.x2 == 210 and r.y2 == 154
    edge = pad_rect(Rect(600, 330, 40, 30), 0.5, W, H)
    assert edge.x2 <= W and edge.y2 <= H


def test_layout_corner_inset_when_not_flush() -> None:
    var = np.full((H, W), 200.0, dtype=np.float32)
    mean = np.zeros((H, W), dtype=np.float32)
    layout = classify_layout(Rect(400, 200, 160, 140), W, H, var, mean, CFG)
    assert layout.kind is SourceLayout.CORNER_INSET
    assert layout.main_keep_rect(W, H) is None


def test_layout_side_panel_when_flush_and_uniform() -> None:
    var = np.full((H, W), 200.0, dtype=np.float32)
    mean = np.random.default_rng(0).uniform(0, 255, (H, W)).astype(np.float32)
    # a flat, static strip on the right
    var[:, 480:] = 2.0
    mean[:, 480:] = 40.0
    layout = classify_layout(Rect(486, 100, 154, 200), W, H, var, mean, CFG)
    assert layout.kind is SourceLayout.SIDE_PANEL
    assert layout.edge == "right"
    keep = layout.main_keep_rect(W, H)
    assert keep is not None
    assert keep.x == 0 and keep.x2 == 486


def test_layout_side_panel_when_flush_and_full_height() -> None:
    layout = classify_layout(Rect(0, 10, 150, 340), W, H, None, None, CFG)
    assert layout.kind is SourceLayout.SIDE_PANEL
    assert layout.edge == "left"


def test_layout_flush_but_busy_is_corner_inset() -> None:
    var = np.full((H, W), 200.0, dtype=np.float32)
    mean = np.random.default_rng(1).uniform(0, 255, (H, W)).astype(np.float32)
    layout = classify_layout(Rect(486, 100, 154, 200), W, H, var, mean, CFG)
    assert layout.kind is SourceLayout.CORNER_INSET


def _pip_stats() -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """Moving programme everywhere, with a static, uniform window at x 300..560, y 100..300."""
    var = np.full((H, W), 200.0, dtype=np.float32)
    mean = np.random.default_rng(2).uniform(0, 255, (H, W)).astype(np.float32)
    var[100:300, 300:560] = 1.0
    mean[100:300, 300:560] = 90.0
    return var, mean


def test_grow_window_reaches_the_pip_border() -> None:
    var, mean = _pip_stats()
    person = Rect(340, 130, 160, 140)
    grown = grow_window(person, var, mean, W, H, CFG)
    assert abs(grown.x - 300) <= CFG.window_band
    assert abs(grown.x2 - 560) <= CFG.window_band
    assert abs(grown.y - 100) <= CFG.window_band
    assert abs(grown.y2 - 300) <= CFG.window_band
    assert grown.w % 2 == 0 and grown.h % 2 == 0


def test_grow_window_is_capped_relative_to_the_box() -> None:
    var, mean = _pip_stats()
    person = Rect(400, 180, 40, 40)  # 1.5x growth cap = 60 px each side
    grown = grow_window(person, var, mean, W, H, CFG)
    assert grown.x >= 340 - CFG.window_band
    assert grown.x2 <= 500 + CFG.window_band


def test_grow_window_is_capped_absolutely_by_frame_width() -> None:
    # Everything static and one uniform grey: nothing but the caps can stop growth.
    var = np.full((H, W), 1.0, dtype=np.float32)
    mean = np.full((H, W), 90.0, dtype=np.float32)
    person = Rect(200, 140, 240, 200)  # 1.5x would allow 360 px per side
    grown = grow_window(person, var, mean, W, H, CFG)
    assert grown.w <= int(W * CFG.window_max_frame_frac) + 2 * CFG.window_band


def test_grow_window_follows_a_drifting_backdrop_but_not_a_step() -> None:
    var = np.full((H, W), 1.0, dtype=np.float32)
    # Backdrop brightens gently to the left, then the programme starts at x<200.
    mean = np.zeros((H, W), dtype=np.float32)
    for x in range(W):
        mean[:, x] = 90.0 + (300 - x) * 0.02 if x >= 200 else 30.0
    person = Rect(360, 150, 120, 120)
    grown = grow_window(person, var, mean, W, H, CFG)
    assert grown.x <= 260  # followed the drift well past the box
    assert grown.x >= 200 - 2 * CFG.window_band  # but stopped at the step


def test_grow_window_does_not_grow_into_moving_picture() -> None:
    var = np.full((H, W), 200.0, dtype=np.float32)
    mean = np.full((H, W), 90.0, dtype=np.float32)
    person = Rect(340, 130, 160, 140)
    grown = grow_window(person, var, mean, W, H, CFG)
    assert grown == person


def test_grow_window_stops_at_a_step_in_mean() -> None:
    var = np.full((H, W), 1.0, dtype=np.float32)  # everything static
    mean = np.full((H, W), 200.0, dtype=np.float32)
    mean[100:300, 300:560] = 90.0  # the window is darker than the static programme
    person = Rect(340, 130, 160, 140)
    grown = grow_window(person, var, mean, W, H, CFG)
    assert abs(grown.x - 300) <= CFG.window_band
    assert abs(grown.x2 - 560) <= CFG.window_band


def test_box_motion_uses_upper_region() -> None:
    diff = np.zeros((100, 100), dtype=np.float32)
    diff[0:30, 40:60] = 10.0  # motion in the top of the box only
    box = Box(40, 0, 60, 100)
    assert box_motion(diff, box, 0.6) == pytest.approx(10.0 * 30 / 60)
    # a box entirely outside the frame has no region to measure
    assert math.isnan(box_motion(diff, Box(200, 200, 260, 260), 0.6))
