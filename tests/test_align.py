"""The aligner: human words on a machine clock.

Every case is a small transcript, a small set of recognised words with times,
and the cues that must come out. The failure modes that matter are silent —
a drifting caption, a cue that runs for eighteen seconds, a dropped sentence —
so each is pinned here.
"""

from __future__ import annotations

from itertools import pairwise

from stm_pipeline.align import (
    MAX_CUE_S,
    MAX_EVENT_S,
    MAX_LINE_CHARS,
    align,
    machine_cues,
    match,
    normalise,
    segment,
    time_words,
    to_webvtt,
    tokenise,
)
from stm_pipeline.asr import TimedWord
from stm_pipeline.captions_vtt import parse_vtt
from stm_pipeline.official_report import Contribution


def spoken(text: str, start: float = 0.0, wps: float = 2.0) -> list[TimedWord]:
    """Recognised words at a steady rate, for building cases."""
    out = []
    t = start
    for w in text.split():
        out.append(TimedWord(w, t, t + 1 / wps * 0.9, 0.9))
        t += 1 / wps
    return out


# --- normalise ------------------------------------------------------------------


def test_normalise_ignores_case_and_punctuation_but_keeps_inner_apostrophes() -> None:
    assert normalise("Minister’s,") == "minister's"
    assert normalise("“Hear,") == "hear"
    assert normalise("2024.") == "2024"
    assert normalise("—") == ""


# --- tokenise --------------------------------------------------------------------


def test_stage_directions_are_single_tokens() -> None:
    words = tokenise([Contribution("A", ["Welcome to the gallery. [Applause.] Next question."])])
    texts = [w.text for w in words]
    assert texts == ["Welcome", "to", "the", "gallery.", "[Applause.]", "Next", "question."]
    assert [w.is_event for w in words] == [False] * 4 + [True] + [False] * 2


# --- match -------------------------------------------------------------------------


def test_matches_through_a_misheard_name_and_an_edited_sentence() -> None:
    report = tokenise(
        [
            Contribution(
                "PO",
                ["welcoming to the gallery Her Excellency Sanja Štiglic, ambassador of Slovenia."],
            )
        ]
    )
    heard = spoken(
        "welcoming to the gallery her Excellency Sanya Stieglitz ambassador of Slovenia "
        "the first leader is Malcolm"
    )
    m = match(report, heard)
    assert m[0] is not None and m[0].text == "welcoming"
    assert m[5] is not None and m[5].text == "Excellency"
    assert m[6] is None and m[7] is None  # the name, misheard
    assert m[8] is not None and m[8].text == "ambassador"
    assert m[-1] is not None and m[-1].text == "Slovenia"


def test_a_repeated_phrase_matches_each_copy_to_its_own_utterance() -> None:
    # The First Minister was interrupted and began the sentence again; the
    # recogniser misheard one word each time. Greedy longest-block matching
    # pinned the second copy to the first utterance. The optimum must not.
    contributions = [
        Contribution(
            "FM", ["a member of the Government that inflicted austerity on this country."]
        ),
        Contribution("PO", ["Mr Kerr, let us hear the First Minister."]),
        Contribution(
            "FM",
            [
                "Malcolm Offord was a supporter of a Government "
                "that inflicted austerity on this country. They represent traditions."
            ],
        ),
    ]
    heard = (
        spoken("a member of a Government that inflicted austerity on this country", start=0.0)
        + spoken("excuse me let us hear the First Minister Mr Kea", start=8.0)
        + spoken(
            "Malcolm Offord was a supporter of a Government that brought the inflicted "
            "austerity on this country it represents",
            start=14.0,
        )
    )
    report = tokenise(contributions)
    m = match(report, heard)
    first = next(i for i, w in enumerate(report) if w.text == "Government")
    second = next(i for i, w in enumerate(report) if w.text == "Government" and i > first)
    a, b = m[first], m[second]
    assert a is not None and b is not None
    assert a.start < 8.0 and b.start > 14.0
    _, summary = align(contributions, heard)
    assert summary.unheard == 0


def test_unheard_words_are_counted_not_smeared() -> None:
    # The Report prints the title in full; the member said "the Secretary has".
    report = [Contribution("A", ["I asked the Secretary for Health and Care has said so."])]
    heard = spoken("I asked the Secretary has said so")
    _, summary = align(report, heard)
    assert summary.unheard == 4


def test_events_never_match_speech() -> None:
    report = tokenise([Contribution("PO", ["Thank you. [Applause.] Thank you."])])
    heard = spoken("thank you applause thank you")
    m = match(report, heard)
    assert m[2] is None


# --- time_words -------------------------------------------------------------------


def test_unmatched_words_share_the_gap_between_anchors() -> None:
    report = tokenise([Contribution("A", ["one two three four five"])])
    heard = [TimedWord("one", 0.0, 0.4), TimedWord("five", 4.0, 4.4)]
    timed = time_words(report, match(report, heard))
    assert [t.anchored for t in timed] == [True, False, False, False, True]
    # two, three, four sit strictly inside (0.4, 4.0) in order
    inner = timed[1:4]
    assert inner[0].start >= 0.4 and inner[-1].end <= 4.0
    assert all(a.end <= b.start + 1e-9 for a, b in pairwise(inner))


def test_words_before_the_first_anchor_are_never_negative() -> None:
    report = tokenise([Contribution("A", ["a b c d e f g h i j k l m n o p q r s t anchor"])])
    heard = [TimedWord("anchor", 1.0, 1.3)]
    timed = time_words(report, match(report, heard))
    assert timed[0].start >= 0.0
    assert timed[-1].anchored


def test_no_anchors_at_all_still_times_every_word() -> None:
    report = tokenise([Contribution("A", ["hello there"])])
    timed = time_words(report, match(report, spoken("completely different")))
    assert len(timed) == 2 and timed[0].start == 0.0 and timed[1].end > timed[0].end


# --- segment -----------------------------------------------------------------------


def test_a_stage_direction_gets_its_own_short_cue_not_the_whole_gap() -> None:
    # The Report cut the procedural sentence that lived in the gap; the gap is 18 s.
    contributions = [
        Contribution("PO", ["of Slovenia. [Applause.]"]),
        Contribution(
            "Malcolm Offord (West Scotland) (Reform)", ["When John Swinney became First Minister."]
        ),
    ]
    heard = [
        TimedWord("of", 19.0, 19.2),
        TimedWord("Slovenia", 19.3, 20.0),
        TimedWord("when", 38.4, 38.6),
        TimedWord("John", 39.0, 39.3),
        TimedWord("Swinney", 39.4, 39.7),
        TimedWord("became", 39.7, 40.0),
        TimedWord("First", 40.0, 40.3),
        TimedWord("Minister", 40.3, 40.7),
    ]
    cues, _ = align(contributions, heard)
    ev = next(c for c in cues if c.text == "[Applause.]")
    # Centred in the 20.0-38.4 gap: nothing says where in it the applause fell.
    assert 24.0 <= ev.start_s <= 30.0
    assert ev.duration_s <= MAX_EVENT_S + 0.01
    assert ev.end_s <= 38.4


def test_speech_the_editor_cut_clings_to_the_anchor_before_it() -> None:
    # "two … seven" were said straight after "one"; the 30 s gap is other speech
    # the Report does not carry. The words must not be smeared across it.
    report = tokenise([Contribution("A", ["one two three four five six seven eight"])])
    heard = [TimedWord("one", 0.0, 0.4), TimedWord("eight", 30.0, 30.4)]
    timed = time_words(report, match(report, heard))
    assert timed[6].end < 5.0
    assert timed[1].start >= 0.4


def test_speech_after_a_stage_direction_leads_into_the_next_anchor() -> None:
    contributions = [
        Contribution("PO", ["Slovenia. [Applause.]"]),
        Contribution("Offord", ["Presiding Officer, it is a pleasure."]),
    ]
    heard = [
        TimedWord("Slovenia", 19.3, 20.0),
        TimedWord("a", 30.0, 30.2),
        TimedWord("pleasure", 30.3, 30.8),
    ]
    cues, _ = align(contributions, heard)
    ev = next(c for c in cues if c.text == "[Applause.]")
    lead = next(c for c in cues if "Presiding Officer" in c.text)
    assert lead.start_s >= 27.0 and lead.start_s < 30.0
    assert ev.end_s <= lead.start_s + 1e-9


def test_no_cue_hangs_longer_than_the_cap() -> None:
    report = [Contribution("A", ["session. Next question is from Douglas Ross."])]
    heard = [
        TimedWord("session", 10.0, 10.4),
        TimedWord("Douglas", 40.0, 40.3),
        TimedWord("Ross", 40.3, 40.6),
    ]
    cues, _ = align(report, heard)
    assert all(c.duration_s <= MAX_CUE_S + 1.0 + 1e-9 for c in cues)


def test_lines_are_balanced_within_the_limit() -> None:
    text = "he promised to lead a delivery-focused, unifying Government that is focused on four"
    contributions = [Contribution("A", [text + "."])]
    cues, _ = align(contributions, spoken(text))
    for c in cues:
        assert all(len(line) <= MAX_LINE_CHARS for line in c.text.split("\n")), c.text


def test_speaker_is_named_on_a_change_of_speaker_in_plain_text() -> None:
    contributions = [
        Contribution("The Presiding Officer (Kenneth Gibson)", ["We now move to questions."]),
        Contribution(
            "1. Malcolm Offord (West Scotland) (Reform)",
            ["When John Swinney became First Minister."],
        ),
    ]
    heard = spoken("we now move to questions when john swinney became first minister")
    cues, _ = align(contributions, heard)
    assert cues[0].text.startswith("The Presiding Officer: ")
    assert cues[1].text.startswith("Malcolm Offord: ")
    assert "(West Scotland)" not in cues[1].text and "1." not in cues[1].text


def test_cues_respect_length_and_duration_limits() -> None:
    long = " ".join(["word"] * 60) + "."
    contributions = [Contribution("A", [long])]
    heard = spoken(long.replace(".", ""), wps=1.0)  # slow speech: duration limit bites
    cues, _ = align(contributions, heard)
    for c in cues:
        for line in c.text.split("\n"):
            assert len(line) <= MAX_LINE_CHARS + 3  # a single long token may overhang
        assert c.text.count("\n") <= 1
        assert c.duration_s <= MAX_CUE_S + 1.0


def test_cues_are_ordered_and_do_not_overlap() -> None:
    text = (
        "The first sentence is here. The second one follows it, with a clause; and a third. Done."
    )
    contributions = [Contribution("A", [text])]
    cues, summary = align(
        contributions, spoken(text.replace(",", "").replace(";", "").replace(".", ""))
    )
    assert summary.anchored_fraction > 0.9
    for a, b in pairwise(cues):
        assert a.end_s <= b.start_s + 1e-9
        assert a.index + 1 == b.index
    assert all(c.duration_s >= 0.5 for c in cues)


def test_webvtt_round_trips_through_the_parser() -> None:
    contributions = [Contribution("A", ["Hello there. This is a caption track."])]
    cues, _ = align(contributions, spoken("hello there this is a caption track"))
    vtt = to_webvtt(cues, "Text: Official Report. Timing: aligned automatically.")
    parsed, preamble = parse_vtt(vtt)
    assert len(parsed) == len(cues)
    assert preamble and preamble[0].startswith("NOTE")
    assert parsed[0].text == cues[0].text


def test_machine_cues_break_on_pauses() -> None:
    words = spoken("one two three") + spoken("four five six", start=10.0)
    cues = machine_cues(words)
    assert len(cues) == 2
    assert cues[1].start_s == 10.0


def test_segment_alone_handles_empty_input() -> None:
    assert segment([], []) == []
