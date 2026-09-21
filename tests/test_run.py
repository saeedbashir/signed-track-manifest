from __future__ import annotations

import json
from pathlib import Path

from stm.model import Rect, SourceLayout
from stm.schema import is_valid, validate_catalogue, validate_title
from stm_pipeline.config import PipelineConfig
from stm_pipeline.detect.mock import ScriptedDetector
from stm_pipeline.harness.ground_truth import GroundTruth
from stm_pipeline.harness.score import score
from stm_pipeline.ingest import SourceEntry
from stm_pipeline.run import build_catalogue, decide, process_title
from stm_pipeline.spike import format_cluster_table, spike
from stm_pipeline.types import Box
from tests.conftest import requires_ffmpeg

# The synthetic clip is 320x180. Script a stable small "signer" bottom-right and a big
# static "anchor" in the middle. testsrc2 moves everywhere, so motion ratios are ~1 and
# the choice rests on size, persistence and stability.
SIGNER = Box(220, 100, 300, 176, 0.9)
ANCHOR = Box(40, 10, 200, 176, 0.95)


def _detector() -> ScriptedDetector:
    return ScriptedDetector(lambda i: [ANCHOR, SIGNER])


def _entry(
    clip: Path,
    *,
    entry_id: str = "synthetic-01",
    captions: str | None = None,
    crop_rect: Rect | None = None,
    layout: SourceLayout | None = None,
) -> SourceEntry:
    return SourceEntry(
        id=entry_id,
        title="Synthetic clip",
        url="https://www.example.gov/synthetic.mp4",
        publisher="Example",
        licence="cc0-1.0",
        attribution="Example.",
        local_path=clip,
        captions=captions,
        crop_rect=crop_rect,
        layout=layout,
    )


@requires_ffmpeg
def test_decide_picks_scripted_signer(synthetic_clip: Path) -> None:
    d = decide(synthetic_clip, _detector(), PipelineConfig())
    assert d.chosen is not None
    assert d.rect_source.iou(SIGNER.to_rect()) > 0.6
    assert d.layout.kind is SourceLayout.CORNER_INSET
    assert 0.0 <= d.confidence <= 1.0
    assert d.analysis is not None and d.analysis.sampled_count == 10
    table = format_cluster_table(d.ranked, d.chosen.cluster_id)
    assert "*" in table


@requires_ffmpeg
def test_process_title_end_to_end(synthetic_clip: Path, tmp_path: Path) -> None:
    captions = tmp_path / "caps.srt"
    captions.write_text("1\n00:00:00,000 --> 00:00:01,000\nHello\n", encoding="utf-8")
    entry = _entry(synthetic_clip, captions=str(captions))
    out = tmp_path / "out"
    manifest_path = process_title(entry, _detector(), PipelineConfig(), out, tmp_path / "work")
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert is_valid(validate_title(data))
    track = data["signLanguage"][0]
    assert track["provenance"] == "human-interpreter"
    assert track["extraction"]["method"] == "detected-crop"
    assert track["extraction"]["fillMethod"] == "interpolated-surround"
    assert (out / "synthetic-01" / track["track"]["url"]).exists()
    assert (out / "synthetic-01" / "main.mp4").exists()
    assert (out / "synthetic-01" / "poster.jpg").exists()
    assert (out / "synthetic-01" / "analysis.json").exists()
    assert data["captions"][0]["kind"] == "verbatim"
    assert data["durationMs"] > 1500

    cat = build_catalogue([manifest_path], out / "catalogue.json")
    cat_data = json.loads(cat.read_text(encoding="utf-8"))
    assert is_valid(validate_catalogue(cat_data))
    assert cat_data["titles"][0]["manifestUrl"] == "synthetic-01/manifest.json"


@requires_ffmpeg
def test_manual_rect_side_panel(synthetic_clip: Path, tmp_path: Path) -> None:
    entry = _entry(
        synthetic_clip,
        entry_id="manual-01",
        crop_rect=Rect(240, 0, 80, 180),
        layout=SourceLayout.SIDE_PANEL,
    )
    manifest_path = process_title(
        entry, _detector(), PipelineConfig(), tmp_path / "out", tmp_path / "work"
    )
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    ex = data["signLanguage"][0]["extraction"]
    assert ex["method"] == "manual-crop"
    assert ex["sourceLayout"] == "side-panel"
    assert ex["fillMethod"] == "none"
    assert ex["confidence"] == 1.0
    assert data["video"]["main"]["width"] == 240


@requires_ffmpeg
def test_harness_scores_against_ground_truth(synthetic_clip: Path) -> None:
    gt = [GroundTruth(clip=synthetic_clip, rect=SIGNER.to_rect(), face=Rect(230, 105, 40, 30))]
    report = score(gt, _detector(), PipelineConfig(), threshold=0.5)
    assert len(report.scores) == 1
    assert report.scores[0].iou > 0.5
    assert report.scores[0].face_contained is True
    assert report.passed == 1
    assert "pass at IoU" in report.table()


@requires_ffmpeg
def test_spike_writes_watchable_outputs(synthetic_clip: Path, tmp_path: Path) -> None:
    result = spike(synthetic_clip, _detector(), PipelineConfig(), tmp_path, detections_video=True)
    assert result.outputs["box"].exists()
    assert result.outputs["fill:interpolated-surround"].exists()
    assert result.outputs["detections"].exists()
    assert result.outputs["report"].exists()
