from __future__ import annotations

import pytest

from stm.model import (
    Caption,
    CaptionKind,
    Extraction,
    ExtractionMethod,
    FillMethod,
    Provenance,
    Rect,
    SignTrack,
    SourceLayout,
    VideoAsset,
)


def _extraction() -> Extraction:
    return Extraction(
        ExtractionMethod.DETECTED_CROP,
        SourceLayout.CORNER_INSET,
        0.9,
        Rect(10, 10, 100, 100),
        FillMethod.INTERPOLATED_SURROUND,
    )


def test_rect_geometry() -> None:
    a = Rect(0, 0, 10, 10)
    b = Rect(5, 5, 10, 10)
    assert a.area == 100
    assert a.iou(b) == pytest.approx(25 / 175)
    assert a.iou(a) == 1.0
    assert a.contains(Rect(2, 2, 4, 4))
    assert not a.contains(b)


def test_rect_rejects_degenerate() -> None:
    with pytest.raises(ValueError, match="positive"):
        Rect(0, 0, 0, 10)
    with pytest.raises(ValueError, match="non-negative"):
        Rect(-1, 0, 5, 5)


def test_generated_caption_is_never_reviewed() -> None:
    cap = Caption.generated(CaptionKind.SIMPLIFIED, "en", "s.vtt", "amazon-bedrock")
    assert cap.reviewed is False
    assert cap.reviewed_by is None
    assert cap.to_dict()["reviewed"] is False


def test_caption_with_generator_defaults_reviewed_false() -> None:
    cap = Caption(CaptionKind.SIMPLIFIED, "en", "s.vtt", generated_by="x")
    assert cap.reviewed is False


def test_reviewed_requires_named_reviewer() -> None:
    with pytest.raises(ValueError, match="reviewed_by"):
        Caption(CaptionKind.SIMPLIFIED, "en", "s.vtt", generated_by="x", reviewed=True)
    cap = Caption.generated(CaptionKind.SIMPLIFIED, "en", "s.vtt", "x")
    with pytest.raises(ValueError, match="reviewer"):
        cap.mark_reviewed("   ")
    reviewed = cap.mark_reviewed("A. Reviewer")
    assert reviewed.reviewed is True
    assert reviewed.reviewed_by == "A. Reviewer"
    assert cap.reviewed is False  # original untouched


def test_reviewed_by_without_reviewed_is_rejected() -> None:
    with pytest.raises(ValueError, match="only meaningful"):
        Caption(CaptionKind.VERBATIM, "en", "v.vtt", reviewed_by="someone")


def test_human_track_requires_extraction() -> None:
    track = VideoAsset("s.mp4", 100, 100)
    with pytest.raises(ValueError, match="extracted"):
        SignTrack("ase", "ASL", Provenance.HUMAN_INTERPRETER, track)
    with pytest.raises(ValueError, match="generatedBy"):
        SignTrack(
            "ase",
            "ASL",
            Provenance.HUMAN_INTERPRETER,
            track,
            extraction=_extraction(),
            generated_by="x",
        )


def test_synthesised_track_requires_generator_and_no_extraction() -> None:
    track = VideoAsset("s.mp4", 100, 100)
    with pytest.raises(ValueError, match="generated"):
        SignTrack("ase", "ASL", Provenance.SYNTHESISED, track)
    with pytest.raises(ValueError, match="no extraction"):
        SignTrack(
            "ase",
            "ASL",
            Provenance.SYNTHESISED,
            track,
            generated_by="avatar",
            extraction=_extraction(),
        )
    ok = SignTrack("ase", "ASL", Provenance.SYNTHESISED, track, generated_by="avatar")
    assert ok.to_dict()["provenance"] == "synthesised"


def test_language_must_be_iso_639_3() -> None:
    track = VideoAsset("s.mp4", 100, 100)
    with pytest.raises(ValueError, match="ISO 639-3"):
        SignTrack("en", "English?", Provenance.HUMAN_INTERPRETER, track, extraction=_extraction())


def test_extraction_confidence_bounds() -> None:
    with pytest.raises(ValueError, match=r"0\.\.1"):
        Extraction(
            ExtractionMethod.DETECTED_CROP,
            SourceLayout.CORNER_INSET,
            1.2,
            Rect(0, 0, 1, 1),
            FillMethod.NONE,
        )
    assert _extraction().low_confidence is False
