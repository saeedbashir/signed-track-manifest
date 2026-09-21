"""WebVTT parsing and rebuilding.

Kept separate from the simplifier so cue handling can be tested without any
network or model. A caption track is the second half of this project's access
argument, so the rule here is that timings are never invented: whatever
rewrites the text, the cues it lands in are the cues the source had.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace

_TIMING = re.compile(
    r"^(?P<start>\d{1,3}:\d{2}:\d{2}[.,]\d{3}|\d{1,2}:\d{2}[.,]\d{3})\s*-->\s*"
    r"(?P<end>\d{1,3}:\d{2}:\d{2}[.,]\d{3}|\d{1,2}:\d{2}[.,]\d{3})"
    r"(?P<settings>.*)$"
)


def parse_timestamp(value: str) -> float:
    """Seconds from a WebVTT timestamp. Accepts mm:ss.mmm and hh:mm:ss.mmm."""
    text = value.strip().replace(",", ".")
    parts = text.split(":")
    if len(parts) == 2:
        hours, minutes, seconds = 0.0, float(parts[0]), float(parts[1])
    elif len(parts) == 3:
        hours, minutes, seconds = float(parts[0]), float(parts[1]), float(parts[2])
    else:
        raise ValueError(f"not a WebVTT timestamp: {value!r}")
    return hours * 3600 + minutes * 60 + seconds


def format_timestamp(seconds: float) -> str:
    """hh:mm:ss.mmm, the form every WebVTT parser accepts."""
    if seconds < 0:
        raise ValueError(f"negative timestamp: {seconds}")
    ms_total = round(seconds * 1000)
    hours, rem = divmod(ms_total, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, ms = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{ms:03d}"


@dataclass(frozen=True)
class Cue:
    """One caption cue. ``index`` is its 1-based position in the file."""

    index: int
    start_s: float
    end_s: float
    text: str
    identifier: str | None = None
    settings: str = ""

    @property
    def duration_s(self) -> float:
        return max(0.0, self.end_s - self.start_s)

    @property
    def timing_line(self) -> str:
        line = f"{format_timestamp(self.start_s)} --> {format_timestamp(self.end_s)}"
        return f"{line}{self.settings}" if self.settings else line

    def with_text(self, text: str) -> Cue:
        return replace(self, text=text)


def parse_vtt(vtt: str) -> tuple[list[Cue], list[str]]:
    """Split WebVTT into cues and the non-cue blocks (NOTE, STYLE, REGION) that precede them.

    Returns ``(cues, preamble_blocks)``. Blocks that are not cues are preserved
    so a rebuilt file keeps styling the source provided.
    """
    text = vtt.replace("\r\n", "\n").replace("\r", "\n")
    blocks = [b for b in text.split("\n\n") if b.strip()]
    cues: list[Cue] = []
    preamble: list[str] = []
    for block in blocks:
        lines = block.split("\n")
        timing_at = next((i for i, line in enumerate(lines) if _TIMING.match(line.strip())), None)
        if timing_at is None:
            stripped = block.strip()
            if not stripped.startswith("WEBVTT"):
                preamble.append(stripped)
            continue
        match = _TIMING.match(lines[timing_at].strip())
        assert match is not None
        identifier = "\n".join(lines[:timing_at]).strip() or None
        body = "\n".join(lines[timing_at + 1 :]).strip()
        cues.append(
            Cue(
                index=len(cues) + 1,
                start_s=parse_timestamp(match.group("start")),
                end_s=parse_timestamp(match.group("end")),
                text=body,
                identifier=identifier,
                settings=match.group("settings").rstrip(),
            )
        )
    return cues, preamble


def build_vtt(cues: list[Cue], preamble: list[str] | None = None) -> str:
    """Render cues back to WebVTT, keeping every original timing."""
    parts = ["WEBVTT", ""]
    for block in preamble or []:
        parts.extend([block, ""])
    for cue in cues:
        if cue.identifier:
            parts.append(cue.identifier)
        parts.append(cue.timing_line)
        parts.append(cue.text)
        parts.append("")
    return "\n".join(parts).rstrip() + "\n"


def word_budget(duration_s: float, words_per_second: float = 2.0, minimum: int = 3) -> int:
    """How many words fit in a cue at a comfortable reading rate.

    Verbatim caption speed routinely exceeds many deaf viewers' reading rate,
    which is the whole reason a simplified track exists. Two words per second is
    a conservative target; the budget is advisory, given to the model as a hint.
    """
    return max(minimum, int(duration_s * words_per_second))
