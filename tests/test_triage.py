"""Triage ranks candidates by demo quality. It never makes a licence judgement."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

from stm.model import Rect, SourceLayout
from stm_pipeline.analyze import Analysis
from stm_pipeline.crop import Layout
from stm_pipeline.run import Decision
from stm_pipeline.triage import (
    Triage,
    format_table,
    measure_background,
    suggest_sources_entry,
    triage,
)
from stm_pipeline.types import FrameInfo
from tests.conftest import requires_ffmpeg


def _analysis(mean: npt.NDArray[np.float64], var: npt.NDArray[np.float64]) -> Analysis:
    h, w = mean.shape
    return Analysis(
        info=FrameInfo(width=w, height=h, fps=25.0, duration_s=6.0, codec="h264"),
        detector_name="scripted",
        sample_every=5,
        sample_scale=1.0,
        sample_width=w,
        sample_height=h,
        frames=[],
        mean_gray=mean.astype(np.float32),
        temporal_var=var.astype(np.float32),
    )


def _decision(
    analysis: Analysis, rect: Rect, kind: SourceLayout = SourceLayout.CORNER_INSET
) -> Decision:
    return Decision(
        rect_source=rect,
        rect_sample=rect,
        person_rect_sample=rect,
        layout=Layout(kind),
        method=None,  # type: ignore[arg-type]
        confidence=0.9,
        chosen=None,
        ranked=[],
        analysis=analysis,
    )


def test_a_calm_background_is_an_easy_fill() -> None:
    mean = np.full((360, 640), 90.0)
    var = np.full((360, 640), 1.0)
    a = _analysis(mean, var)
    rect = Rect(400, 200, 120, 120)
    detail, motion, difficulty = measure_background(_decision(a, rect), rect)
    assert detail < 1.0
    assert motion < 5.0
    assert difficulty < 0.1


def test_a_busy_moving_background_is_a_hard_fill() -> None:
    rng = np.random.default_rng(0)
    mean = rng.uniform(0, 255, (360, 640))
    var = np.full((360, 640), 400.0)
    a = _analysis(mean, var)
    rect = Rect(400, 200, 120, 120)
    detail, _motion, difficulty = measure_background(_decision(a, rect), rect)
    assert detail > 50.0
    assert difficulty > 0.9


def test_the_window_itself_is_excluded_from_the_measurement() -> None:
    """A busy window on a calm surround must still read as an easy fill:
    interpolation copies from the surroundings, never from what was removed."""
    mean = np.full((360, 640), 90.0)
    rng = np.random.default_rng(1)
    mean[200:320, 400:520] = rng.uniform(0, 255, (120, 120))
    var = np.full((360, 640), 1.0)
    a = _analysis(mean, var)
    rect = Rect(400, 200, 120, 120)
    _, _, difficulty = measure_background(_decision(a, rect), rect)
    assert difficulty < 0.2


@pytest.mark.parametrize(
    ("difficulty", "confidence", "layout", "expected"),
    [
        (0.0, 0.9, str(SourceLayout.SIDE_PANEL), "excellent"),
        (0.9, 0.9, str(SourceLayout.SIDE_PANEL), "excellent"),
        (0.2, 0.9, str(SourceLayout.CORNER_INSET), "good"),
        (0.5, 0.9, str(SourceLayout.CORNER_INSET), "marginal"),
        (0.8, 0.9, str(SourceLayout.CORNER_INSET), "poor"),
        (0.1, 0.5, str(SourceLayout.CORNER_INSET), "marginal"),
    ],
)
def test_verdicts(difficulty: float, confidence: float, layout: str, expected: str) -> None:
    r = Triage(
        clip="c.mp4", ok=True, layout=layout, confidence=confidence, fill_difficulty=difficulty
    )
    assert r.verdict == expected


def test_a_failed_clip_is_unusable_and_ranks_last() -> None:
    bad = Triage(clip="x.mp4", ok=False, error="no signer")
    good = Triage(clip="y.mp4", ok=True, layout=str(SourceLayout.CORNER_INSET), fill_difficulty=0.1)
    assert bad.verdict == "unusable"
    assert sorted([bad, good], key=lambda r: r.rank)[0] is good


def test_the_table_says_what_it_does_not_measure() -> None:
    out = format_table([Triage(clip="a.mp4", ok=True, layout=str(SourceLayout.CORNER_INSET))])
    assert "Licence" in out
    assert "your call" in out


def test_the_sources_skeleton_leaves_every_judgement_blank() -> None:
    r = Triage(
        clip="/tmp/WH Briefing.mp4",
        ok=True,
        layout=str(SourceLayout.CORNER_INSET),
        confidence=0.9,
        fill_difficulty=0.2,
        duration_s=540,
    )
    entry = suggest_sources_entry(r)
    assert "id: wh-briefing" in entry
    for judgement in ("licence", "production_credit", "acquisition_route", "attribution"):
        assert f'{judgement}: ""' in entry
    assert "clean, low detail" in entry
    assert "duration_s: 540" in entry


@requires_ffmpeg
def test_triage_reports_unusable_rather_than_raising(tmp_path: Path) -> None:
    from stm_pipeline.config import PipelineConfig
    from stm_pipeline.detect.mock import ScriptedDetector

    missing = tmp_path / "nope.mp4"
    r = triage(missing, ScriptedDetector({}), PipelineConfig())
    assert not r.ok
    assert r.verdict == "unusable"
    assert r.error
