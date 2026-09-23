"""The Scottish Parliament's Official Report, read into contributions.

One publisher's markup, kept on its own so that another publisher's transcript
page is another reader, not a change to the aligner. The aligner takes
:class:`Contribution` objects and does not know where they came from.

The Report is *edited verbatim*: false starts and repetitions are removed and
grammar is tidied, while stage directions such as ``[Applause.]`` are kept.
That is exactly the text a caption viewer wants, and the reason the aligner
has to tolerate words the speaker said that the page does not carry.

The markup, as published in September 2026: each contribution begins with
``<p id="orscontributions_…">`` holding the speaker in ``<strong><a>``, followed
by plain ``<p>`` paragraphs until the next one; ``<p class="lead">`` are
sub-headings; a bare ``<p>14:01</p>`` is a clock. The page footer arrives as
ordinary paragraphs, so the reader stops at the first one that is plainly
footer.
"""

from __future__ import annotations

import html
import re
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

_CONTRIBUTION = re.compile(r'<p id="orscontributions_[^"]*">(.*?)</p>', re.S)
_PARAGRAPH = re.compile(r"<p(?P<attrs>[^>]*)>(?P<body>.*?)</p>", re.S)
_CLASS = re.compile(r'class="([^"]*)"')
_SPEAKER = re.compile(r"<strong>(.*?)</strong>", re.S)
_TAG = re.compile(r"<[^>]+>")
_CLOCK = re.compile(r"^\d{1,2}:\d{2}$")
_FOOTER = ("Back to Official Report", "©", "&copy;")
#: Where the item's text ends and the page's own furniture begins.
_END_MARKERS = ("<!-- Previous/Next navigation -->", '<p class="h5">')
#: A leading question number, as in "1. Malcolm Offord (West Scotland) (Reform)".
_QUESTION_NUMBER = re.compile(r"^\d+\.\s+")


@dataclass
class Contribution:
    """What one speaker said, as the Report prints it."""

    speaker: str
    paragraphs: list[str] = field(default_factory=list)
    #: The clock the Report printed just before this contribution, if any ("14:01").
    clock: str | None = None

    @property
    def display_name(self) -> str:
        """The name a caption shows on a change of speaker.

        The Report writes "1. Malcolm Offord (West Scotland) (Reform)" and
        "The First Minister (John Swinney)". A caption has room for neither the
        constituency nor the parenthesis, and the office is what matters on
        screen — so "Malcolm Offord" and "The First Minister".
        """
        name = _QUESTION_NUMBER.sub("", self.speaker).strip()
        cut = name.find(" (")
        return name[:cut].strip() if cut > 0 else name

    @property
    def word_count(self) -> int:
        return sum(len(p.split()) for p in self.paragraphs)


def _clean(fragment: str) -> str:
    return html.unescape(_TAG.sub("", fragment)).replace("\xa0", " ").strip()


def _is_footer(text: str) -> bool:
    return any(text.startswith(f) or f in text for f in _FOOTER)


def parse_official_report(page: str) -> list[Contribution]:
    """Contributions from an Official Report item page (or any fragment of one)."""
    start = page.find('<p id="orscontributions_')
    if start < 0:
        return []
    body = page[start:]
    for marker in _END_MARKERS:
        cut = body.find(marker)
        if cut > 0:
            body = body[:cut]
    contributions: list[Contribution] = []
    current: Contribution | None = None
    pending_clock: str | None = None
    for m in _PARAGRAPH.finditer(body):
        attrs = m.group("attrs")
        # One page printed the speaker's colon at the start of the speech
        # paragraph ("<p><span>: I have always…"); the colon is not speech.
        text = _clean(m.group("body")).lstrip(": ").strip()
        if 'id="orscontributions_' in attrs:
            # The speaker is the <strong> element; the share widget beside it is not.
            strong = _SPEAKER.search(m.group("body"))
            speaker = _clean(strong.group(1)) if strong else text
            current = Contribution(speaker=speaker, clock=pending_clock)
            pending_clock = None
            contributions.append(current)
            continue
        if not text:
            continue
        if _is_footer(text):
            break
        if _CLOCK.match(text):
            pending_clock = text
            continue
        cls = _CLASS.search(attrs)
        if cls and "lead" in cls.group(1).split():
            # A sub-heading ("Party Leaders", "Government Priorities"): not speech.
            continue
        if current is not None:
            current.paragraphs.append(text)
    # The anchor paragraph at the top of an item has no speaker and no text.
    return [c for c in contributions if c.speaker and c.paragraphs]


def fetch_official_report(source: str, cache_dir: Path | None = None) -> str:
    """The page HTML from a URL or a local file. A URL is cached beside the work when asked."""
    if "://" not in source:
        return Path(source).read_text(encoding="utf-8")
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cached = cache_dir / "official-report.html"
        if cached.exists():
            return cached.read_text(encoding="utf-8")
    req = urllib.request.Request(
        source,
        headers={
            "User-Agent": "signed-track-manifest/0.2 (+https://github.com/saeedbashir/signed-track-manifest)"
        },
    )
    with urllib.request.urlopen(req, timeout=60) as response:
        text: str = response.read().decode("utf-8", errors="replace")
    if cache_dir is not None:
        (cache_dir / "official-report.html").write_text(text, encoding="utf-8")
    return text
