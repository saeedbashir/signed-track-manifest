"""Thin, typed wrapper over the ffmpeg and ffprobe binaries.

All media transforms go through here so they are mockable and so every filter
graph is written down in one place. Binaries are resolved from ``STM_FFMPEG``
and ``STM_FFPROBE`` when set, otherwise from PATH.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from collections.abc import Sequence
from fractions import Fraction
from pathlib import Path

from stm.model import FillMethod, Rect
from stm_pipeline.types import FrameInfo


class FfmpegError(RuntimeError):
    """ffmpeg or ffprobe exited non-zero."""


def ffmpeg_bin() -> str:
    return os.environ.get("STM_FFMPEG", "ffmpeg")


def ffprobe_bin() -> str:
    return os.environ.get("STM_FFPROBE", "ffprobe")


def available() -> bool:
    return shutil.which(ffmpeg_bin()) is not None and shutil.which(ffprobe_bin()) is not None


def _run(binary: str, args: Sequence[str]) -> str:
    cmd = [binary, *args]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise FfmpegError(f"{' '.join(cmd)}\n{proc.stderr.strip()}")
    return proc.stdout


def run_ffmpeg(args: Sequence[str]) -> None:
    _run(ffmpeg_bin(), ["-hide_banner", "-loglevel", "error", "-y", *args])


def probe(path: Path) -> FrameInfo:
    """Read dimensions, frame rate, duration and codec of the first video stream."""
    out = _run(
        ffprobe_bin(),
        [
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,r_frame_rate,avg_frame_rate,codec_name,nb_frames,duration",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(path),
        ],
    )
    data = json.loads(out)
    streams = data.get("streams") or []
    if not streams:
        raise FfmpegError(f"no video stream in {path}")
    s = streams[0]
    rate = s.get("avg_frame_rate") or s.get("r_frame_rate") or "0/1"
    try:
        fps = float(Fraction(rate))
    except (ValueError, ZeroDivisionError):
        fps = 0.0
    duration = s.get("duration") or data.get("format", {}).get("duration") or "0"
    nb = s.get("nb_frames")
    return FrameInfo(
        width=int(s["width"]),
        height=int(s["height"]),
        fps=fps,
        duration_s=float(duration),
        codec=str(s.get("codec_name", "unknown")),
        frame_count=int(nb) if nb not in (None, "N/A") else None,
    )


def _even(n: int) -> int:
    return n if n % 2 == 0 else n - 1


def crop(
    src: Path,
    rect: Rect,
    dst: Path,
    max_height: int | None = 540,
    crf: int = 23,
    keep_audio: bool = False,
) -> None:
    """Cut ``rect`` out of the video, optionally scaling down to ``max_height``. H.264."""
    filters = [f"crop={_even(rect.w)}:{_even(rect.h)}:{rect.x}:{rect.y}"]
    if max_height is not None and rect.h > max_height:
        filters.append(f"scale=-2:{_even(max_height)}")
    audio = ["-c:a", "aac", "-b:a", "96k"] if keep_audio else ["-an"]
    run_ffmpeg(
        [
            "-i",
            str(src),
            "-vf",
            ",".join(filters),
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            str(crf),
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            *audio,
            str(dst),
        ]
    )


def _delogo_safe(rect: Rect, width: int, height: int, margin: int = 2) -> Rect:
    """ffmpeg's delogo refuses rectangles touching the frame edge; inset by ``margin``."""
    x = max(margin, rect.x)
    y = max(margin, rect.y)
    x2 = min(width - margin, rect.x2)
    y2 = min(height - margin, rect.y2)
    return Rect(x, y, max(1, x2 - x), max(1, y2 - y))


def fill(
    src: Path,
    rect: Rect,
    dst: Path,
    width: int,
    height: int,
    method: FillMethod = FillMethod.INTERPOLATED_SURROUND,
    blur_sigma: float = 6.0,
    crf: int = 20,
) -> None:
    """Patch the vacated rectangle in the main video. Never generative.

    ``interpolated-surround`` uses ffmpeg's ``delogo`` (interpolates inward from the
    surrounding pixels) then softens the patch with a Gaussian blur so the
    interpolation banding does not show. ``blurred-surround`` simply blurs the
    region in place, which leaves a smudge of the interpreter and exists for
    side-by-side comparison. ``none`` remuxes untouched.
    """
    if method is FillMethod.NONE:
        run_ffmpeg(["-i", str(src), "-c", "copy", "-movflags", "+faststart", str(dst)])
        return

    r = _delogo_safe(rect, width, height)
    if method is FillMethod.INTERPOLATED_SURROUND:
        graph = (
            f"[0:v]delogo=x={r.x}:y={r.y}:w={r.w}:h={r.h}[d];"
            f"[d]split[a][b];"
            f"[b]crop={r.w}:{r.h}:{r.x}:{r.y},gblur=sigma={blur_sigma}[p];"
            f"[a][p]overlay={r.x}:{r.y}[v]"
        )
    else:
        graph = (
            f"[0:v]split[a][b];"
            f"[b]crop={r.w}:{r.h}:{r.x}:{r.y},gblur=sigma={blur_sigma * 3:.1f}[p];"
            f"[a][p]overlay={r.x}:{r.y}[v]"
        )
    run_ffmpeg(
        [
            "-i",
            str(src),
            "-filter_complex",
            graph,
            "-map",
            "[v]",
            "-map",
            "0:a?",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            str(crf),
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "copy",
            "-movflags",
            "+faststart",
            str(dst),
        ]
    )


def crop_away(src: Path, keep: Rect, dst: Path, crf: int = 20) -> None:
    """Re-frame the main video to ``keep`` (side-panel case: drop the interpreter strip)."""
    run_ffmpeg(
        [
            "-i",
            str(src),
            "-vf",
            f"crop={_even(keep.w)}:{_even(keep.h)}:{keep.x}:{keep.y}",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            str(crf),
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "copy",
            "-movflags",
            "+faststart",
            str(dst),
        ]
    )


def poster(src: Path, time_s: float, dst: Path) -> None:
    """Write a single JPEG frame at ``time_s``."""
    run_ffmpeg(["-ss", f"{time_s:.3f}", "-i", str(src), "-frames:v", "1", "-q:v", "2", str(dst)])


def drawbox(src: Path, boxes: Sequence[tuple[Rect, str, int]], dst: Path) -> None:
    """Burn rectangle outlines (rect, colour, thickness) onto the video in one pass."""
    if not boxes:
        raise ValueError("drawbox needs at least one rectangle")
    filters = ",".join(
        f"drawbox=x={r.x}:y={r.y}:w={r.w}:h={r.h}:color={colour}@1:t={t}" for r, colour, t in boxes
    )
    run_ffmpeg(
        [
            "-i",
            str(src),
            "-vf",
            filters,
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "23",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "copy",
            str(dst),
        ]
    )


def synthetic_clip(
    dst: Path, width: int = 320, height: int = 180, seconds: float = 2.0, fps: int = 25
) -> None:
    """Generate a small moving test pattern. Used by tests and by nothing else."""
    run_ffmpeg(
        [
            "-f",
            "lavfi",
            "-i",
            f"testsrc2=size={width}x{height}:rate={fps}",
            "-t",
            f"{seconds}",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-pix_fmt",
            "yuv420p",
            str(dst),
        ]
    )
