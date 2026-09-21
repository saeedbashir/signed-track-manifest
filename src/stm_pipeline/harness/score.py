from __future__ import annotations

import json
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from stm_pipeline.config import PipelineConfig
from stm_pipeline.detect.base import Detector
from stm_pipeline.harness.ground_truth import GroundTruth
from stm_pipeline.run import NoSignerFoundError, decide

PASS_IOU = 0.85


@dataclass(frozen=True)
class ClipScore:
    clip: str
    iou: float
    passed: bool
    face_contained: bool | None
    confidence: float | None
    layout: str | None
    seconds: float
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "clip": self.clip,
            "iou": round(self.iou, 4),
            "passed": self.passed,
            "faceContained": self.face_contained,
            "confidence": None if self.confidence is None else round(self.confidence, 4),
            "layout": self.layout,
            "seconds": round(self.seconds, 1),
            "error": self.error,
        }


@dataclass(frozen=True)
class Report:
    threshold: float
    scores: list[ClipScore]

    @property
    def passed(self) -> int:
        return sum(1 for s in self.scores if s.passed)

    @property
    def mean_iou(self) -> float:
        return sum(s.iou for s in self.scores) / len(self.scores) if self.scores else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "threshold": self.threshold,
            "clips": len(self.scores),
            "passed": self.passed,
            "meanIou": round(self.mean_iou, 4),
            "scores": [s.to_dict() for s in self.scores],
        }

    def table(self) -> str:
        lines = [
            f"{'clip':40} {'iou':>6} {'pass':>5} {'face':>5} {'conf':>5} {'layout':13} {'s':>6}"
        ]
        for s in self.scores:
            face = "-" if s.face_contained is None else ("yes" if s.face_contained else "NO")
            conf = "-" if s.confidence is None else f"{s.confidence:.2f}"
            name = s.clip if len(s.clip) <= 40 else "…" + s.clip[-39:]
            lines.append(
                f"{name:40} {s.iou:6.3f} {'ok' if s.passed else 'FAIL':>5} {face:>5} {conf:>5} "
                f"{s.layout or '-':13} {s.seconds:6.1f}" + (f"  {s.error}" if s.error else "")
            )
        lines.append(
            f"\n{self.passed}/{len(self.scores)} pass at IoU >= {self.threshold}; "
            f"mean IoU {self.mean_iou:.3f}"
        )
        return "\n".join(lines)

    def write_json(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")


def score(
    items: Sequence[GroundTruth],
    detector: Detector,
    config: PipelineConfig,
    threshold: float = PASS_IOU,
) -> Report:
    """Run the locate step on every clip and compare against the hand-drawn rectangle."""
    scores: list[ClipScore] = []
    for gt in items:
        t0 = time.monotonic()
        try:
            d = decide(gt.clip, detector, config)
        except (NoSignerFoundError, OSError) as exc:
            scores.append(
                ClipScore(
                    str(gt.clip), 0.0, False, None, None, None, time.monotonic() - t0, str(exc)
                )
            )
            continue
        iou = d.rect_source.iou(gt.rect)
        face = d.rect_source.contains(gt.face) if gt.face is not None else None
        scores.append(
            ClipScore(
                clip=str(gt.clip),
                iou=iou,
                passed=iou >= threshold and face is not False,
                face_contained=face,
                confidence=d.confidence,
                layout=str(d.layout.kind),
                seconds=time.monotonic() - t0,
            )
        )
    return Report(threshold, scores)
