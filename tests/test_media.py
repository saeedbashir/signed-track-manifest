from __future__ import annotations

from pathlib import Path

import pytest

from stm.model import FillMethod, Rect
from stm_pipeline import ffmpeg
from stm_pipeline.sample import iter_sampled_frames
from tests.conftest import requires_ffmpeg


@requires_ffmpeg
def test_probe(synthetic_clip: Path) -> None:
    info = ffmpeg.probe(synthetic_clip)
    assert (info.width, info.height) == (320, 180)
    assert info.fps == pytest.approx(25.0)
    assert 1.9 <= info.duration_s <= 2.1
    assert info.codec == "h264"


@requires_ffmpeg
def test_sampling_every_fifth_frame(synthetic_clip: Path) -> None:
    frames = list(iter_sampled_frames(synthetic_clip, every=5, max_width=640))
    assert len(frames) == 10
    assert [f.index for f in frames[:3]] == [0, 5, 10]
    assert frames[0].scale == 1.0
    assert frames[0].image.shape == (180, 320, 3)


@requires_ffmpeg
def test_sampling_downscales(synthetic_clip: Path) -> None:
    frames = list(iter_sampled_frames(synthetic_clip, every=10, max_width=160))
    assert frames[0].image.shape[1] == 160
    assert frames[0].scale == pytest.approx(0.5)


@requires_ffmpeg
def test_crop_fill_poster_drawbox(synthetic_clip: Path, tmp_path: Path) -> None:
    rect = Rect(200, 100, 100, 70)
    signer = tmp_path / "signer.mp4"
    ffmpeg.crop(synthetic_clip, rect, signer, max_height=540)
    assert ffmpeg.probe(signer).width == 100

    for method in (FillMethod.INTERPOLATED_SURROUND, FillMethod.BLURRED_SURROUND, FillMethod.NONE):
        out = tmp_path / f"main-{method}.mp4"
        ffmpeg.fill(synthetic_clip, rect, out, 320, 180, method)
        assert ffmpeg.probe(out).width == 320

    poster = tmp_path / "poster.jpg"
    ffmpeg.poster(synthetic_clip, 0.5, poster)
    assert poster.stat().st_size > 0

    boxed = tmp_path / "boxed.mp4"
    ffmpeg.drawbox(synthetic_clip, [(rect, "red", 4), (Rect(10, 10, 40, 40), "orange", 2)], boxed)
    assert ffmpeg.probe(boxed).height == 180


@requires_ffmpeg
def test_crop_away_side_panel(synthetic_clip: Path, tmp_path: Path) -> None:
    out = tmp_path / "main.mp4"
    ffmpeg.crop_away(synthetic_clip, Rect(0, 0, 240, 180), out)
    assert ffmpeg.probe(out).width == 240


@requires_ffmpeg
def test_crop_scales_down_tall_rects(synthetic_clip: Path, tmp_path: Path) -> None:
    out = tmp_path / "s.mp4"
    ffmpeg.crop(synthetic_clip, Rect(0, 0, 100, 180), out, max_height=90)
    assert ffmpeg.probe(out).height == 90


def test_ffmpeg_error_is_raised(tmp_path: Path) -> None:
    if not ffmpeg.available():
        pytest.skip("ffmpeg not on PATH")
    with pytest.raises(ffmpeg.FfmpegError):
        ffmpeg.probe(tmp_path / "missing.mp4")
