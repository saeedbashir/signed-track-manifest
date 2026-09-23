"""Put a human-written transcript on a machine's clock.

The Official Report says what was said, in the words a human editor settled
on; the recogniser says *when*, to the word, in whatever words it heard. This
module matches the two and gives every word of the human text a time, then
cuts the result into caption cues.

Three rules, each learned from the first real session:

* **Anchor on matches, interpolate across the rest.** In the first three minutes
  of a First Minister's Questions, 78 % of recognised words matched the Report
  exactly. The other 22 % were a Slovenian ambassador's name, "2024" heard as
  "24", a sentence the editor cut, and applause. Words that match are anchors
  with exact times; words between anchors share the gap in proportion to their
  length. Nothing is dropped and nothing is invented: every Report word gets a
  time, and the text is the Report's throughout.
* **A stage direction owns its gap, up to a point.** ``[Applause.]`` has no
  speech to match. It sits between the anchor before and the anchor after, and
  the Report editor has usually removed the procedural sentence that also lived
  there, so the gap can be long. The cue takes the start of the gap and a few
  seconds, not all of it: applause is information, an eighteen-second caption
  saying so is not.
* **Cues are read, not just displayed.** Two lines of forty-two characters, one
  to six seconds, cut at sentence ends first and clause marks second, the
  speaker named on a change of speaker in plain text — the player strips markup,
  and a viewer who cannot hear the voice change needs the name.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from itertools import pairwise

from rapidfuzz.distance import LCSseq

from stm_pipeline.asr import TimedWord
from stm_pipeline.captions_vtt import Cue, build_vtt
from stm_pipeline.official_report import Contribution

#: Layout limits for a cue on a television, read from about three metres.
MAX_LINE_CHARS = 42
MAX_LINES = 2
MAX_CUE_S = 6.0
MIN_CUE_S = 1.0
#: Gaps shorter than this are closed, so the plate does not flicker between cues.
CLOSE_GAP_S = 0.2
#: A stage direction shows for at most this long, however long its gap.
MAX_EVENT_S = 5.0
#: Words per second assumed where there is no anchor on one side.
FALLBACK_WPS = 2.5
#: A stage direction weighs this many characters when a gap is shared out.
EVENT_WEIGHT = 12
#: No spoken word lasts longer than this; a recogniser that hands the following
#: pause to the word before it is corrected here, so a cue ends when speech does.
MAX_WORD_S = 1.0
#: A silence this long inside a paragraph ends the cue, because the speaker stopped.
PAUSE_S = 1.5
#: How long after the last word before the room reacts.
EVENT_LAG_S = 0.4

_EVENT = re.compile(r"\[[^\]]*\]")
_SENTENCE_END = re.compile(r"[.!?][\"'”’)\]]*$")
_CLAUSE_END = re.compile(r"[,;:—–][\"'”’)\]]*$")


def normalise(word: str) -> str:
    """The form two words are compared in: lower-case, letters, digits and inner apostrophes."""
    w = word.lower().replace("’", "'").replace("‘", "'")
    w = re.sub(r"[^a-z0-9']+", "", w)
    return w.strip("'")


@dataclass(frozen=True)
class ReportWord:
    """One token of the human text, with where it came from."""

    text: str
    contribution: int
    paragraph: int
    is_event: bool = False

    @property
    def norm(self) -> str:
        # An event can never match a spoken word; give it a key speech cannot produce.
        return "\x00event" if self.is_event else normalise(self.text)

    @property
    def weight(self) -> int:
        return EVENT_WEIGHT if self.is_event else max(1, len(self.text))


def tokenise(contributions: list[Contribution]) -> list[ReportWord]:
    """The Report as a flat word list, stage directions kept as single tokens."""
    words: list[ReportWord] = []
    for ci, c in enumerate(contributions):
        for pi, para in enumerate(c.paragraphs):
            pos = 0
            for m in _EVENT.finditer(para):
                for tok in para[pos : m.start()].split():
                    words.append(ReportWord(tok, ci, pi))
                words.append(ReportWord(m.group(0), ci, pi, is_event=True))
                pos = m.end()
            for tok in para[pos:].split():
                words.append(ReportWord(tok, ci, pi))
    return words


def match(report: list[ReportWord], spoken: list[TimedWord]) -> list[TimedWord | None]:
    """For each Report word, the recognised word it matches, or None.

    A true longest common subsequence of the normalised forms, from rapidfuzz.
    The standard library's ``difflib`` was tried first and is not good enough
    here: it takes the longest matching block greedily, and a chamber repeats
    itself — a member interrupted mid-sentence starts the sentence again — so
    the second copy of a phrase was pinned to the first utterance whenever one
    misheard word made that block a word longer. The optimum prefers matching
    both copies, and in a 34-minute session it put 50 more words back where
    they were said.
    """
    a = [w.norm for w in report]
    b = [normalise(w.text) for w in spoken]
    out: list[TimedWord | None] = [None] * len(report)
    for block in LCSseq.editops(a, b).as_matching_blocks():
        for k in range(block.size):
            if a[block.a + k]:
                out[block.a + k] = spoken[block.b + k]
    return out


@dataclass(frozen=True)
class TimedReportWord:
    word: ReportWord
    start: float
    end: float
    anchored: bool


def _natural_seconds(words: list[ReportWord]) -> float:
    """How long these words take to say at an ordinary rate."""
    return sum(w.weight for w in words) / 5.0 / FALLBACK_WPS


def time_words(report: list[ReportWord], matches: list[TimedWord | None]) -> list[TimedReportWord]:
    """Every Report word with a time: matched words exactly, the rest by rule.

    The rule for a run of unmatched words between two anchors comes from what
    such a run *is*. Mostly it is speech the editor tidied — a false start, a
    misheard name — and it was said right where the anchors put it, so when the
    gap is roomy the words cling to the anchor before them at an ordinary rate
    rather than being smeared across the silence. Speech that follows a stage
    direction leads into the next anchor instead, so it clings to that. And a
    stage direction has no speech at all: it begins as the speech before it
    ends and runs for at most :data:`MAX_EVENT_S`.
    """
    n = len(report)
    if n == 0:
        return []
    anchors = [i for i, m in enumerate(matches) if m is not None]
    timed: list[TimedReportWord | None] = [None] * n
    for i in anchors:
        m = matches[i]
        assert m is not None
        end = min(max(m.end, m.start + 0.05), m.start + MAX_WORD_S)
        timed[i] = TimedReportWord(report[i], m.start, end, True)

    def spread(lo: int, hi: int, t0: float, t1: float) -> None:
        """report[lo:hi] across [t0, t1] by weight, in order."""
        total = sum(report[k].weight for k in range(lo, hi)) or 1
        span = max(t1 - t0, 0.05 * (hi - lo))
        at = t0
        for k in range(lo, hi):
            dur = span * report[k].weight / total
            timed[k] = TimedReportWord(report[k], at, at + dur, False)
            at += dur

    def place(lo: int, hi: int, t0: float, t1: float) -> None:
        """A run of unmatched words between times t0 and t1.

        Up to the first stage direction or change of speaker the words are the
        previous speaker's, said straight after the anchor before them; from
        there on they lead into the anchor after. Stage directions take the
        middle of what is left.
        """
        prev_c = report[lo - 1].contribution
        split = next(
            (k for k in range(lo, hi) if report[k].is_event or report[k].contribution != prev_c),
            hi,
        )
        before = list(range(lo, split))
        after = [k for k in range(split, hi) if not report[k].is_event]
        events = [k for k in range(lo, hi) if report[k].is_event]
        if not events and not after:
            need = _natural_seconds([report[k] for k in before])
            if t1 - t0 > 2 * need + 1.0:
                spread(lo, hi, t0, t0 + need)  # cling to the speech before
            else:
                spread(lo, hi, t0, t1)
            return
        nb = _natural_seconds([report[k] for k in before])
        na = _natural_seconds([report[k] for k in after])
        if t1 - t0 < nb + na + 1.0 * len(events):
            spread(lo, hi, t0, t1)  # cramped: in order, sharing what there is
            return
        b_end, a_start = t0 + nb, t1 - na
        if before:
            spread(before[0], before[-1] + 1, t0, b_end)
        if after:
            spread(after[0], after[-1] + 1, a_start, t1)
        if events:
            # Applause and interruptions begin the moment the speech before them
            # ends — checked against the picture: the chamber is clapping within
            # a second of "Slovenia." — so an event starts there, not in the
            # middle of whatever silence follows it.
            window = min(MAX_EVENT_S, max(1.0, (a_start - b_end) / len(events)))
            start = b_end + EVENT_LAG_S
            for k in events:
                end = min(start + window, a_start) if a_start > start else start + 1.0
                timed[k] = TimedReportWord(report[k], start, max(end, start + 1.0), False)
                start = end

    if not anchors:
        spread(0, n, 0.0, _natural_seconds(report))
        return [t for t in timed if t is not None]

    first, last = anchors[0], anchors[-1]
    if first > 0:
        head = timed[first]
        assert head is not None
        need = _natural_seconds(report[:first])
        spread(0, first, max(0.0, head.start - need), head.start)
    for lo, hi in pairwise(anchors):
        if hi - lo > 1:
            a, b = timed[lo], timed[hi]
            assert a is not None and b is not None
            place(lo + 1, hi, a.end, b.start)
    if last < n - 1:
        tail = timed[last]
        assert tail is not None
        spread(last + 1, n, tail.end, tail.end + _natural_seconds(report[last + 1 :]))
    return [t for t in timed if t is not None]


def _break_lines(text: str) -> str:
    """Two lines at most, split at the space that leaves the longer line shortest.

    Splitting at the middle character put 39 on one line and 43 on the other
    for an 83-character cue; the limit is per line, so the split that matters
    is the one that minimises the longer half.
    """
    if len(text) <= MAX_LINE_CHARS:
        return text
    candidates = [i for i, ch in enumerate(text) if ch == " "]
    if not candidates:
        return text
    mid = len(text) // 2
    cut = min(candidates, key=lambda i: (max(i, len(text) - i - 1), abs(i - mid)))
    return text[:cut].rstrip() + "\n" + text[cut + 1 :].lstrip()


def _fits(text: str) -> bool:
    """Whether the text reads within the line limits once broken."""
    lines = _break_lines(text).split("\n")
    return len(lines) <= MAX_LINES and all(len(line) <= MAX_LINE_CHARS for line in lines)


def segment(timed: list[TimedReportWord], contributions: list[Contribution]) -> list[Cue]:
    """Cut timed words into cues a viewer can read."""
    cues: list[Cue] = []
    buf: list[TimedReportWord] = []
    max_chars = MAX_LINE_CHARS * MAX_LINES

    def prefix(first: TimedReportWord) -> str:
        if first.word.paragraph == 0 and _is_first_in_contribution(first, timed):
            return f"{contributions[first.word.contribution].display_name}: "
        return ""

    def flush() -> None:
        if not buf:
            return
        first = buf[0]
        text = prefix(first) + " ".join(t.word.text for t in buf)
        cues.append(Cue(len(cues) + 1, first.start, buf[-1].end, _break_lines(text)))
        buf.clear()

    for i, t in enumerate(timed):
        if t.word.is_event:
            flush()
            cues.append(Cue(len(cues) + 1, t.start, max(t.end, t.start + 1.0), t.word.text))
            continue
        nxt = timed[i + 1] if i + 1 < len(timed) else None
        ends_unit = bool(_SENTENCE_END.search(t.word.text)) or (
            nxt is None
            or nxt.word.is_event
            or (nxt.word.contribution, nxt.word.paragraph)
            != (t.word.contribution, t.word.paragraph)
        )
        if buf and (
            t.word.contribution != buf[0].word.contribution
            or t.word.paragraph != buf[0].word.paragraph
        ):
            flush()
        if buf:
            prospective = prefix(buf[0]) + " ".join(x.word.text for x in buf) + " " + t.word.text
            too_long = not _fits(prospective)
            too_slow = t.end - buf[0].start > MAX_CUE_S
            paused = t.start - buf[-1].end > PAUSE_S
            carry: list[TimedReportWord] = []
            if (too_long or too_slow) and not paused and ends_unit and len(buf) >= 4:
                # "…wee day trip to" / "Wales?" reads badly; the last word of the
                # full cue comes down to keep the sentence's end company.
                carry.append(buf.pop())
                if len(carry[0].word.text) <= 3 and len(buf) >= 4:
                    carry.insert(0, buf.pop())
            if too_long or too_slow or paused:
                flush()
                buf.extend(carry)
        buf.append(t)
        joined = " ".join(x.word.text for x in buf)
        if (_SENTENCE_END.search(t.word.text) and len(joined) >= 20) or (
            _CLAUSE_END.search(t.word.text) and len(joined) >= max_chars * 0.6
        ):
            flush()
    flush()
    return _tidy(cues)


def _is_first_in_contribution(t: TimedReportWord, timed: list[TimedReportWord]) -> bool:
    for x in timed:
        if x.word.contribution == t.word.contribution and not x.word.is_event:
            return x is t
    return False


def _tidy(cues: list[Cue]) -> list[Cue]:
    """Minimum durations, closed gaps, no overlaps, strictly increasing indexes."""
    out: list[Cue] = []
    prev_end = 0.0
    for i, c in enumerate(cues):
        # Never before the cue before it: a squeezed run keeps its order and its text.
        start, end = max(c.start_s, prev_end), c.end_s
        nxt = cues[i + 1].start_s if i + 1 < len(cues) else None
        if end - start < MIN_CUE_S:
            end = start + MIN_CUE_S
        # A caption left hanging over a gap the Report edited out is a stale caption.
        end = min(end, start + MAX_CUE_S + 1.0)
        if nxt is not None:
            if nxt - end < CLOSE_GAP_S:
                end = nxt
            end = min(end, nxt)
        if end <= start:
            # A collision the source made; keep order, give this cue what room there is.
            end = start + (0.5 if nxt is None or start + 0.5 <= nxt else 0.05)
        out.append(Cue(len(out) + 1, round(start, 3), round(end, 3), c.text))
        prev_end = end
    return out


@dataclass(frozen=True)
class AlignmentReport:
    """What happened, for the analysis file and the person checking the result."""

    report_words: int
    spoken_words: int
    anchored: int
    events: int
    cues: int
    #: Report words in runs the recogniser left no time for — an editor's expansion
    #: ("NHS" printed as "national health service"), a heckle the microphone
    #: missed. They flash past where they belong; a person checking the track
    #: should know how many there are.
    unheard: int = 0

    @property
    def anchored_fraction(self) -> float:
        speech = self.report_words - self.events
        return self.anchored / speech if speech else 0.0

    def to_dict(self) -> dict[str, float | int]:
        return {
            "reportWords": self.report_words,
            "spokenWords": self.spoken_words,
            "anchored": self.anchored,
            "anchoredFraction": round(self.anchored_fraction, 3),
            "events": self.events,
            "cues": self.cues,
            "unheard": self.unheard,
        }


def align(
    contributions: list[Contribution], spoken: list[TimedWord]
) -> tuple[list[Cue], AlignmentReport]:
    """The Report's words as caption cues on the recogniser's clock."""
    report = tokenise(contributions)
    matches = match(report, spoken)
    timed = time_words(report, matches)
    cues = segment(timed, contributions)
    summary = AlignmentReport(
        report_words=len(report),
        spoken_words=len(spoken),
        anchored=sum(1 for t in timed if t.anchored),
        events=sum(1 for w in report if w.is_event),
        cues=len(cues),
        unheard=_unheard(timed),
    )
    return cues, summary


#: Faster than anyone speaks: a run placed at this rate had no speech to sit on.
_UNHEARD_WPS = 8.0


def _unheard(timed: list[TimedReportWord]) -> int:
    """Speech words in unanchored runs of three or more that got no real time."""
    count = 0
    i = 0
    while i < len(timed):
        if timed[i].anchored or timed[i].word.is_event:
            i += 1
            continue
        j = i
        while j < len(timed) and not timed[j].anchored and not timed[j].word.is_event:
            j += 1
        span = max(timed[j - 1].end - timed[i].start, 1e-3)
        if j - i >= 3 and (j - i) / span > _UNHEARD_WPS:
            count += j - i
        i = j
    return count


def to_webvtt(cues: list[Cue], note: str | None = None) -> str:
    preamble = [f"NOTE {note}"] if note else None
    return build_vtt(cues, preamble)


def machine_cues(spoken: list[TimedWord]) -> list[Cue]:
    """Cues from the recogniser's own words, for content with no written record.

    Cut on the same reading limits, and additionally on a pause: a second of
    silence ends a cue, because the speaker did.
    """
    cues: list[Cue] = []
    buf: list[TimedWord] = []

    def flush() -> None:
        if buf:
            cues.append(
                Cue(
                    len(cues) + 1,
                    buf[0].start,
                    buf[-1].end,
                    _break_lines(" ".join(w.text for w in buf)),
                )
            )
            buf.clear()

    for w in spoken:
        if buf:
            prospective = " ".join(x.text for x in buf) + " " + w.text
            if (
                not _fits(prospective)
                or w.end - buf[0].start > MAX_CUE_S
                or w.start - buf[-1].end > 1.0
            ):
                flush()
        buf.append(w)
        if _SENTENCE_END.search(w.text) and len(" ".join(x.text for x in buf)) >= 20:
            flush()
    flush()
    return _tidy(cues)
