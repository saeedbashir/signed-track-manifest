"""Reading caption files the source provides, and converting them to WebVTT.

Cue parsing lives in :mod:`stm_pipeline.captions_vtt`; simplification lives in
:mod:`stm_pipeline.simplify`. This module is only about getting a file on disk
into WebVTT text.
"""

from __future__ import annotations

import re
from pathlib import Path

_SRT_TIME = re.compile(r"(\d{2}):(\d{2}):(\d{2}),(\d{3})")
_SRT_INDEX = re.compile(r"^\d+\s*$")


def srt_to_vtt(srt: str) -> str:
    """Minimal SRT -> WebVTT: drop cue numbers, swap the comma in timestamps."""
    lines = ["WEBVTT", ""]
    for raw in srt.replace("\r\n", "\n").split("\n"):
        line = raw.rstrip()
        if _SRT_INDEX.match(line):
            continue
        line = _SRT_TIME.sub(r"\1:\2:\3.\4", line)
        lines.append(line)
    text = "\n".join(lines).rstrip() + "\n"
    return text


def read_as_vtt(path: Path) -> str:
    """Load a caption file as WebVTT text. Accepts .vtt and .srt."""
    text = path.read_text(encoding="utf-8-sig")
    if path.suffix.lower() == ".srt":
        return srt_to_vtt(text)
    if not text.lstrip().startswith("WEBVTT"):
        raise ValueError(f"{path} is neither WebVTT nor SRT")
    return text
