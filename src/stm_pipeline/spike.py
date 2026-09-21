"""Experiment 3 tool: run detection on a clip and produce things a human can watch.

Writes ``<clip>.spike.mp4`` with the chosen fixed rectangle drawn on every frame,
``<clip>.fill.<method>.mp4`` with the rectangle patched, and optionally
``<clip>.detections.mp4`` with per-frame boxes at the sample rate. Prints the
cluster table so you can see why it chose what it chose.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2

from stm.model import FillMethod, Rect
from stm_pipeline import ffmpeg
from stm_pipeline.config import PipelineConfig
from stm_pipeline.crop import scale_rect
from stm_pipeline.detect.base import Detector
from stm_pipeline.identify import ClusterFeatures
from stm_pipeline.run import Decision, decide
from stm_pipeline.sample import iter_sampled_frames


@dataclass(frozen=True)
class SpikeResult:
    decision: Decision
    outputs: dict[str, Path]

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision.to_dict(),
            "outputs": {k: str(v) for k, v in self.outputs.items()},
        }


def format_cluster_table(ranked: Sequence[ClusterFeatures], chosen_id: int | None) -> str:
    header = (
        f"{'':2}{'id':>3} {'frames':>6} {'persist':>7} {'stable':>6} {'size':>5} "
        f"{'motion×':>7} {'score':>5}  reason"
    )
    lines = [header]
    for f in ranked:
        mark = "*" if f.cluster_id == chosen_id else " "
        motion = "   n/a" if f.motion_ratio != f.motion_ratio else f"{f.motion_ratio:7.2f}"
        lines.append(
            f"{mark:2}{f.cluster_id:>3} {f.frames:>6} {f.persistence:7.2f} {f.stability:6.2f} "
            f"{f.size_frac:5.2f} {motion} {f.score:5.2f}  {f.reason}"
        )
    return "\n".join(lines)


def _write_detections_video(
    clip: Path, detector: Detector, config: PipelineConfig, decision: Decision, dst: Path
) -> None:
    fps = max(
        1.0, (decision.analysis.info.fps if decision.analysis else 25.0) / config.sample_every
    )
    writer = None
    try:
        for sf in iter_sampled_frames(clip, config.sample_every, config.sample_width):
            frame = sf.image.copy()
            for b in detector.detect(sf):
                cv2.rectangle(
                    frame, (int(b.x1), int(b.y1)), (int(b.x2), int(b.y2)), (0, 200, 255), 1
                )
            if decision.person_rect_sample is not None:
                p = decision.person_rect_sample
                cv2.rectangle(frame, (p.x, p.y), (p.x2, p.y2), (0, 165, 255), 1)
            r = decision.rect_sample
            cv2.rectangle(frame, (r.x, r.y), (r.x2, r.y2), (0, 0, 255), 2)
            if writer is None:
                fourcc = cv2.VideoWriter.fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(dst), fourcc, fps, (frame.shape[1], frame.shape[0]))
            writer.write(frame)
    finally:
        if writer is not None:
            writer.release()


def spike(
    clip: Path,
    detector: Detector,
    config: PipelineConfig,
    out_dir: Path | None = None,
    fill_methods: Sequence[FillMethod] = (FillMethod.INTERPOLATED_SURROUND,),
    detections_video: bool = False,
) -> SpikeResult:
    out_dir = out_dir or clip.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    decision = decide(clip, detector, config)
    info = decision.analysis.info if decision.analysis else ffmpeg.probe(clip)
    stem = clip.stem
    outputs: dict[str, Path] = {}

    box_path = out_dir / f"{stem}.spike.mp4"
    boxes: list[tuple[Rect, str, int]] = [(decision.rect_source, "red", 4)]
    if decision.person_rect_sample is not None and decision.analysis is not None:
        inv = 1.0 / decision.analysis.sample_scale if decision.analysis.sample_scale else 1.0
        person = scale_rect(decision.person_rect_sample, inv, info.width, info.height)
        boxes.append((person, "orange", 2))
    ffmpeg.drawbox(clip, boxes, box_path)
    outputs["box"] = box_path

    for method in fill_methods:
        fill_path = out_dir / f"{stem}.fill.{method}.mp4"
        ffmpeg.fill(
            clip,
            decision.rect_source,
            fill_path,
            info.width,
            info.height,
            method,
            config.fill_blur_sigma,
        )
        outputs[f"fill:{method}"] = fill_path

    if detections_video:
        det_path = out_dir / f"{stem}.detections.mp4"
        _write_detections_video(clip, detector, config, decision, det_path)
        outputs["detections"] = det_path

    report = out_dir / f"{stem}.spike.json"
    result = SpikeResult(decision, outputs)
    report.write_text(json.dumps(result.to_dict(), indent=2) + "\n", encoding="utf-8")
    outputs["report"] = report
    return result
