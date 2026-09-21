"""Which of the detected people is the interpreter?

Detections are clustered by spatial position across time. The signer is the
cluster that is positionally stable, persistent across most frames, small
relative to the frame, and high in hand-motion energy. Motion carries the most
weight: it is what separates an interpreter from a static anchor or a logo.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from stm_pipeline.config import PipelineConfig
from stm_pipeline.types import Box, FrameDetections


@dataclass
class Cluster:
    id: int
    ema: Box
    frame_indices: list[int] = field(default_factory=list)
    boxes: list[Box] = field(default_factory=list)
    motions: list[float] = field(default_factory=list)
    frame_motions: list[float] = field(default_factory=list)

    def add(
        self, frame_index: int, box: Box, motion: float, frame_motion: float, alpha: float
    ) -> None:
        self.frame_indices.append(frame_index)
        self.boxes.append(box)
        self.motions.append(motion)
        self.frame_motions.append(frame_motion)
        e = self.ema
        self.ema = Box(
            e.x1 + alpha * (box.x1 - e.x1),
            e.y1 + alpha * (box.y1 - e.y1),
            e.x2 + alpha * (box.x2 - e.x2),
            e.y2 + alpha * (box.y2 - e.y2),
        )


@dataclass(frozen=True)
class ClusterFeatures:
    cluster_id: int
    frames: int
    persistence: float
    centre_std_norm: float
    stability: float
    size_frac: float
    motion_ratio: float
    motion_score: float
    score: float
    eligible: bool
    reason: str
    mean_box: Box

    def to_dict(self) -> dict[str, Any]:
        return {
            "clusterId": self.cluster_id,
            "frames": self.frames,
            "persistence": round(self.persistence, 4),
            "centreStdNorm": round(self.centre_std_norm, 5),
            "stability": round(self.stability, 4),
            "sizeFrac": round(self.size_frac, 4),
            "motionRatio": None if math.isnan(self.motion_ratio) else round(self.motion_ratio, 3),
            "motionScore": round(self.motion_score, 4),
            "score": round(self.score, 4),
            "eligible": self.eligible,
            "reason": self.reason,
            "meanBox": [
                round(v, 1)
                for v in (self.mean_box.x1, self.mean_box.y1, self.mean_box.x2, self.mean_box.y2)
            ],
        }


def cluster_detections(frames: Sequence[FrameDetections], config: PipelineConfig) -> list[Cluster]:
    """Greedy tracking-by-overlap: each box joins the cluster whose running box it overlaps most."""
    clusters: list[Cluster] = []
    for fd in frames:
        if not fd.boxes:
            continue
        candidates: list[tuple[float, int, int]] = []
        for bi, box in enumerate(fd.boxes):
            for ci, cl in enumerate(clusters):
                iou = box.iou(cl.ema)
                if iou >= config.cluster_iou:
                    candidates.append((iou, bi, ci))
        candidates.sort(reverse=True)
        used_boxes: set[int] = set()
        used_clusters: set[int] = set()
        for _, bi, ci in candidates:
            if bi in used_boxes or ci in used_clusters:
                continue
            clusters[ci].add(
                fd.index, fd.boxes[bi], fd.motions[bi], fd.frame_motion, config.cluster_ema_alpha
            )
            used_boxes.add(bi)
            used_clusters.add(ci)
        for bi, box in enumerate(fd.boxes):
            if bi in used_boxes:
                continue
            cl = Cluster(id=len(clusters), ema=box)
            cl.add(fd.index, box, fd.motions[bi], fd.frame_motion, config.cluster_ema_alpha)
            clusters.append(cl)
    return clusters


def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))


def _nanmean(values: Sequence[float]) -> float:
    arr = np.asarray([v for v in values if not math.isnan(v)], dtype=np.float64)
    return float(arr.mean()) if arr.size else math.nan


def features(
    cluster: Cluster, total_frames: int, frame_w: int, frame_h: int, config: PipelineConfig
) -> ClusterFeatures:
    n = len(cluster.boxes)
    persistence = n / total_frames if total_frames else 0.0

    cxs = np.asarray([b.cx for b in cluster.boxes])
    cys = np.asarray([b.cy for b in cluster.boxes])
    diag = math.hypot(frame_w, frame_h)
    centre_std_norm = float(math.hypot(cxs.std(), cys.std()) / diag) if n > 1 else 0.0
    span = config.stability_zero_at - config.stability_full_marks_at
    stability = _clamp01(1.0 - (centre_std_norm - config.stability_full_marks_at) / span)

    size_frac = float(np.median([b.w for b in cluster.boxes]) / frame_w) if frame_w else 1.0
    size_score = _clamp01(1.0 - size_frac / config.max_size_frac)

    box_motion = _nanmean(cluster.motions)
    frame_motion = _nanmean(cluster.frame_motions)
    if math.isnan(box_motion) or math.isnan(frame_motion):
        motion_ratio = math.nan
        motion_score = 0.0
    else:
        motion_ratio = box_motion / max(frame_motion, 1e-3)
        motion_score = _clamp01(motion_ratio / config.motion_full_marks_ratio)

    w = config.weights
    score = (
        w.motion * motion_score
        + w.persistence * persistence
        + w.stability * stability
        + w.size * size_score
    )

    reasons: list[str] = []
    if persistence < config.min_persistence:
        reasons.append(f"persistence {persistence:.2f} < {config.min_persistence}")
    if size_frac > config.max_size_frac:
        reasons.append(f"size {size_frac:.2f} > {config.max_size_frac}")
    eligible = not reasons

    mean_box = Box(
        float(np.mean([b.x1 for b in cluster.boxes])),
        float(np.mean([b.y1 for b in cluster.boxes])),
        float(np.mean([b.x2 for b in cluster.boxes])),
        float(np.mean([b.y2 for b in cluster.boxes])),
    )
    return ClusterFeatures(
        cluster_id=cluster.id,
        frames=n,
        persistence=persistence,
        centre_std_norm=centre_std_norm,
        stability=stability,
        size_frac=size_frac,
        motion_ratio=motion_ratio,
        motion_score=motion_score,
        score=score,
        eligible=eligible,
        reason="; ".join(reasons) if reasons else "ok",
        mean_box=mean_box,
    )


def rank_clusters(
    clusters: Sequence[Cluster],
    total_frames: int,
    frame_w: int,
    frame_h: int,
    config: PipelineConfig,
) -> list[ClusterFeatures]:
    """All clusters with their features, eligible ones first, best score first."""
    ranked = [features(c, total_frames, frame_w, frame_h, config) for c in clusters]
    ranked.sort(key=lambda f: (f.eligible, f.score), reverse=True)
    return ranked


def choose_signer(ranked: Sequence[ClusterFeatures]) -> ClusterFeatures | None:
    for f in ranked:
        if f.eligible:
            return f
    return None
