"""The signing-activity track: measurement, bucketing, normalisation, and the model."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from stm.model import (
    Activity,
    Extraction,
    ExtractionMethod,
    FillMethod,
    Provenance,
    Rect,
    SignTrack,
    Source,
    SourceLayout,
    TitleManifest,
    VideoAsset,
)
from stm.schema import is_valid, validate_title
from stm_pipeline.activity import (
    bucket_values,
    idle_fraction,
    measure_activity,
    normalise,
    sparkline,
)
from stm_pipeline.config import PipelineConfig
from tests.conftest import requires_ffmpeg

# --- bucketing ----------------------------------------------------------------


def test_buckets_are_means_per_interval_from_zero() -> None:
    samples = [(0.1, 2.0), (0.5, 4.0), (1.2, 10.0), (2.9, 1.0)]
    assert bucket_values(samples, 1.0, 3.0) == [3.0, 10.0, 1.0]


def test_an_empty_interval_repeats_the_previous_value() -> None:
    # Nothing measured is not the same as "still".
    samples = [(0.2, 8.0), (2.2, 2.0)]
    assert bucket_values(samples, 1.0, 3.0) == [8.0, 8.0, 2.0]


def test_a_leading_empty_interval_is_zero() -> None:
    assert bucket_values([(1.5, 5.0)], 1.0, 2.0) == [0.0, 5.0]


def test_duration_sets_the_length_and_late_samples_land_in_the_last_bucket() -> None:
    assert len(bucket_values([], 1.0, 4.4)) == 5
    assert bucket_values([(9.0, 7.0)], 1.0, 2.0) == [0.0, 7.0]


def test_nan_samples_are_ignored() -> None:
    assert bucket_values([(0.1, math.nan), (0.5, 4.0)], 1.0, 1.0) == [4.0]


def test_bad_interval_is_refused() -> None:
    with pytest.raises(ValueError):
        bucket_values([], 0.0, 1.0)


# --- normalisation ------------------------------------------------------------


def test_normalise_scales_to_the_percentile_and_clamps() -> None:
    values, top = normalise([0.0, 5.0, 10.0, 10.0, 10.0, 200.0], percentile=50.0)
    assert top == 10.0
    assert values == [0, 50, 100, 100, 100, 100]


def test_a_still_track_is_all_zeros_not_a_division() -> None:
    values, top = normalise([0.0, 0.0, 0.0])
    assert values == [0, 0, 0] and top == 0.0


def test_normalise_empty() -> None:
    assert normalise([]) == ([], 0.0)


def test_idle_fraction() -> None:
    assert idle_fraction([0, 10, 19, 20, 100]) == pytest.approx(3 / 5)
    assert idle_fraction([]) == 0.0


def test_sparkline_is_one_character_per_value_up_to_width() -> None:
    assert len(sparkline([0, 50, 100])) == 3
    assert sparkline([0, 100]) == " █"
    assert len(sparkline(list(range(0, 101)) * 3, width=40)) == 40
    assert sparkline([]) == ""


# --- the model ----------------------------------------------------------------


def test_activity_at_pads_the_tail_and_zeroes_before_the_start() -> None:
    a = Activity(values=[10, 20, 30], interval_ms=1000)
    assert a.at(-5) == 0
    assert a.at(0) == 10
    assert a.at(999) == 10
    assert a.at(1000) == 20
    assert a.at(2500) == 30
    assert a.at(99_000) == 30


def test_activity_roundtrip() -> None:
    a = Activity(values=[0, 100], interval_ms=500)
    assert Activity.from_dict(json.loads(json.dumps(a.to_dict()))) == a


@pytest.mark.parametrize(
    "kwargs",
    [
        {"values": []},
        {"values": [0, 101]},
        {"values": [0, 1], "interval_ms": 100},
        {"values": [0, 1], "interval_ms": 9000},
        {"values": [0, 1], "scale": "absolute"},
    ],
)
def test_activity_model_rejects_bad_shapes(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        Activity(**kwargs)  # type: ignore[arg-type]


def _track(activity: Activity | None) -> SignTrack:
    return SignTrack(
        language="bfi",
        language_label="British Sign Language",
        provenance=Provenance.HUMAN_INTERPRETER,
        track=VideoAsset("signer-bfi.mp4", 420, 380),
        extraction=Extraction(
            ExtractionMethod.MANUAL_CROP,
            SourceLayout.SIDE_PANEL,
            1.0,
            Rect(1080, 108, 840, 760),
            FillMethod.NONE,
        ),
        activity=activity,
    )


def _manifest(track: SignTrack, version: str) -> TitleManifest:
    return TitleManifest(
        id="t-1",
        title="T",
        duration_ms=3000,
        source=Source("P", "https://p.example", "cc0-1.0", "P."),
        main=VideoAsset("main.mp4", 1080, 1080),
        poster="poster.jpg",
        sign_language=[track],
        stm_version=version,
    )


def test_manifest_with_activity_validates_as_0_2() -> None:
    m = _manifest(_track(Activity(values=[0, 60, 90])), "0.2")
    data = m.to_dict()
    assert data["signLanguage"][0]["activity"]["values"] == [0, 60, 90]
    assert is_valid(validate_title(data))


def test_manifest_without_activity_omits_the_key() -> None:
    data = _manifest(_track(None), "0.2").to_dict()
    assert "activity" not in data["signLanguage"][0]
    assert is_valid(validate_title(data))


def test_model_refuses_activity_on_a_0_1_manifest() -> None:
    with pytest.raises(ValueError, match=r"0\.2"):
        _manifest(_track(Activity(values=[1])), "0.1")


def test_model_refuses_an_unknown_version() -> None:
    with pytest.raises(ValueError, match="stmVersion"):
        _manifest(_track(None), "0.9")


# --- measurement ----------------------------------------------------------------


@requires_ffmpeg
def test_measure_activity_on_the_synthetic_clip(synthetic_clip: Path) -> None:
    # testsrc2 moves everywhere, so any rectangle is active; what this checks is
    # the plumbing — one value per second of the clip, integers in range, the top
    # of the track at 100 by construction, and a positive p95 recorded.
    m = measure_activity(synthetic_clip, Rect(200, 60, 100, 100), PipelineConfig())
    assert m.activity.interval_ms == 1000
    assert len(m.activity.values) == 2
    assert all(0 <= v <= 100 for v in m.activity.values)
    assert max(m.activity.values) == 100
    assert m.p95 > 0 and m.samples > 0
    assert m.to_dict()["length"] == 2


@requires_ffmpeg
def test_measure_activity_interval_can_be_changed(synthetic_clip: Path) -> None:
    m = measure_activity(synthetic_clip, Rect(200, 60, 100, 100), PipelineConfig(), 500)
    assert m.activity.interval_ms == 500
    assert len(m.activity.values) == 4
