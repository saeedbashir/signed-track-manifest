"""Pipeline configuration. Every tunable lives here so the harness can sweep them."""

from __future__ import annotations

from dataclasses import dataclass, field

from stm.model import FillMethod


@dataclass(frozen=True)
class IdentifyWeights:
    """How the four signals combine into a signer score. Motion is weighted heaviest
    because it is what separates an interpreter from a static anchor or a logo."""

    motion: float = 0.40
    persistence: float = 0.25
    stability: float = 0.25
    size: float = 0.10


@dataclass(frozen=True)
class PipelineConfig:
    # Sampling
    sample_every: int = 5
    sample_width: int = 640

    # Detection
    detector: str = "yolox-tiny"
    score_threshold: float = 0.35
    nms_threshold: float = 0.45

    # Motion
    motion_upper_fraction: float = 0.6  # hands and face live in the upper part of a person box

    # Clustering
    cluster_iou: float = 0.3
    cluster_ema_alpha: float = 0.2

    # Eligibility and scoring
    min_persistence: float = 0.5
    max_size_frac: float = 0.45  # box width / frame width; an interpreter is small
    stability_full_marks_at: float = 0.01  # centre std / frame diagonal
    stability_zero_at: float = 0.06
    motion_full_marks_ratio: float = 3.0  # box motion / frame motion
    weights: IdentifyWeights = field(default_factory=IdentifyWeights)

    # Crop
    crop_low_percentile: float = 5.0
    crop_high_percentile: float = 95.0
    crop_pad: float = 0.08

    # Picture-in-picture window growth (corner inset only)
    window_growth: bool = True
    window_var_threshold: float = 40.0  # outside band must be this static (temporal variance)
    # Maximum step in mean grey between consecutive bands. Compared against the
    # band accepted last, not the first, so a backdrop that drifts is followed
    # while a step onto the programme stops growth. Swept on the synthetic
    # corpus: 4 and 6 both classify every layout correctly, 8 lets growth walk
    # off a dark backdrop onto a calm programme. Retune on real clips.
    window_mean_delta: float = 6.0
    # Each side may move out by this fraction of the box. Generous, because a
    # side-panel interpreter can stand in a strip several times their own width.
    window_max_growth: float = 1.5
    # Absolute backstop: a signer window is never most of the frame. This is what
    # bounds growth across a programme that happens to be flat and static, which
    # the per-side cap alone cannot do.
    window_max_frame_frac: float = 0.6
    window_band: int = 2  # pixels per step, at sample resolution
    window_margin: float = 0.03  # outward pad after growth so the window's border line is covered

    # Layout classification
    edge_tolerance: float = 0.02  # fraction of frame width counted as "flush"
    panel_var_threshold: float = 40.0  # mean temporal variance of gray in the panel
    panel_std_threshold: float = 12.0  # spatial std of the mean gray image in the panel
    full_height_fraction: float = 0.85

    # Outputs
    signer_max_height: int = 540
    signer_crf: int = 23
    main_crf: int = 20
    fill_method: FillMethod = FillMethod.INTERPOLATED_SURROUND
    fill_blur_sigma: float = 6.0
    poster_at_fraction: float = 0.25
