from __future__ import annotations

from pathlib import Path

import pytest

from stm_pipeline.captions import read_as_vtt, srt_to_vtt
from stm_pipeline.ingest import BlockedSourceError, SourceError, check_url_allowed, load_sources

SRT = """1
00:00:01,000 --> 00:00:02,500
Hello there.

2
00:00:03,000 --> 00:00:04,000
Second line
continues.
"""


def test_srt_to_vtt() -> None:
    vtt = srt_to_vtt(SRT)
    assert vtt.startswith("WEBVTT\n\n")
    assert "00:00:01.000 --> 00:00:02.500" in vtt
    assert "\n1\n" not in vtt
    assert "continues." in vtt


def test_read_as_vtt_accepts_both(tmp_path: Path) -> None:
    (tmp_path / "a.srt").write_text(SRT, encoding="utf-8")
    (tmp_path / "b.vtt").write_text(
        "WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nHi\n", encoding="utf-8"
    )
    (tmp_path / "c.txt").write_text("not captions", encoding="utf-8")
    assert read_as_vtt(tmp_path / "a.srt").startswith("WEBVTT")
    assert read_as_vtt(tmp_path / "b.vtt").startswith("WEBVTT")
    with pytest.raises(ValueError, match="neither"):
        read_as_vtt(tmp_path / "c.txt")


@pytest.mark.parametrize(
    "url",
    [
        "https://www.youtube.com/watch?v=abc",
        "https://youtu.be/abc",
        "https://rr1---sn.googlevideo.com/videoplayback",
        "https://m.youtube.com/watch?v=abc",
    ],
)
def test_platform_rips_are_refused(url: str) -> None:
    with pytest.raises(BlockedSourceError):
        check_url_allowed(url)


def test_agency_urls_are_allowed() -> None:
    check_url_allowed("https://www.example.gov/media/briefing.mp4")
    check_url_allowed("https://archive.org/download/item/file.mp4")


def test_load_sources(tmp_path: Path) -> None:
    (tmp_path / "sources.yaml").write_text(
        """
sources:
  - id: briefing-01
    title: Briefing one
    url: https://www.example.gov/briefing-01.mp4
    publisher: Example Agency
    licence: public-domain-usgov
    attribution: Example Agency. Public domain.
    production_credit: "verified: in-house"
    acquisition_route: agency download page
    captions: https://www.example.gov/briefing-01.vtt
  - id: manual-02
    title: Manual rect
    url: https://www.example.gov/two.mp4
    publisher: Example Agency
    licence: public-domain-usgov
    attribution: Example Agency.
    crop_rect: {x: 1400, y: 600, w: 400, h: 400}
    layout: corner-inset
""",
        encoding="utf-8",
    )
    entries = load_sources(tmp_path / "sources.yaml")
    assert [e.id for e in entries] == ["briefing-01", "manual-02"]
    assert entries[0].sign_language == "ase"
    assert entries[1].crop_rect is not None and entries[1].crop_rect.w == 400


def test_manual_rect_needs_layout(tmp_path: Path) -> None:
    (tmp_path / "s.yaml").write_text(
        """
sources:
  - id: x-1
    title: t
    url: https://www.example.gov/x.mp4
    publisher: p
    licence: l
    attribution: a
    crop_rect: {x: 0, y: 0, w: 10, h: 10}
""",
        encoding="utf-8",
    )
    with pytest.raises(SourceError, match="layout"):
        load_sources(tmp_path / "s.yaml")


def test_duplicate_ids_rejected(tmp_path: Path) -> None:
    body = """
sources:
  - {id: same-1, title: t, url: https://a.gov/x.mp4, publisher: p, licence: l, attribution: a}
  - {id: same-1, title: t, url: https://a.gov/y.mp4, publisher: p, licence: l, attribution: a}
"""
    (tmp_path / "s.yaml").write_text(body, encoding="utf-8")
    with pytest.raises(SourceError, match="duplicate"):
        load_sources(tmp_path / "s.yaml")
