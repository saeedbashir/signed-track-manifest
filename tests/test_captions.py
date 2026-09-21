"""VTT handling and caption simplification. No network: the client is faked."""

from __future__ import annotations

from typing import Any

import pytest

from stm.model import Caption, CaptionKind
from stm_pipeline.captions_vtt import (
    build_vtt,
    format_timestamp,
    parse_timestamp,
    parse_vtt,
    word_budget,
)
from stm_pipeline.simplify import (
    BedrockSimplifier,
    NullSimplifier,
    Simplifier,
    parse_reply,
    render_chunk,
)

VTT = """WEBVTT

NOTE this is a comment block

1
00:00:01.000 --> 00:00:05.000
The Secretary announced that the department has reached a decision
regarding the application.

00:00:05.500 --> 00:00:08.000 line:90%
[APPLAUSE]

speaker-2
00:00:08.000 --> 00:00:14.000
Approximately 4,200 households in Harris County will not be affected.
"""


def test_timestamp_round_trip() -> None:
    assert parse_timestamp("00:01:02.500") == pytest.approx(62.5)
    assert parse_timestamp("01:02.500") == pytest.approx(62.5)
    assert parse_timestamp("01:00:00,250") == pytest.approx(3600.25)
    assert format_timestamp(62.5) == "00:01:02.500"
    assert format_timestamp(3600.25) == "01:00:00.250"
    with pytest.raises(ValueError, match="timestamp"):
        parse_timestamp("nonsense")


def test_parse_vtt_keeps_identifiers_settings_and_notes() -> None:
    cues, preamble = parse_vtt(VTT)
    assert [c.index for c in cues] == [1, 2, 3]
    assert cues[0].identifier == "1"
    assert cues[0].start_s == pytest.approx(1.0)
    assert cues[0].duration_s == pytest.approx(4.0)
    assert "reached a decision" in cues[0].text
    assert cues[1].settings == " line:90%"
    assert cues[2].identifier == "speaker-2"
    assert preamble == ["NOTE this is a comment block"]


def test_build_vtt_round_trips_timings() -> None:
    cues, preamble = parse_vtt(VTT)
    rebuilt = build_vtt(cues, preamble)
    again, again_preamble = parse_vtt(rebuilt)
    assert rebuilt.startswith("WEBVTT\n")
    assert [c.start_s for c in again] == [c.start_s for c in cues]
    assert [c.end_s for c in again] == [c.end_s for c in cues]
    assert [c.settings for c in again] == [c.settings for c in cues]
    assert again_preamble == preamble


def test_word_budget_scales_with_duration() -> None:
    assert word_budget(5.0, 2.0) == 10
    assert word_budget(0.1, 2.0) == 3  # never below the floor


def test_render_chunk_numbers_and_budgets() -> None:
    cues, _ = parse_vtt(VTT)
    rendered = render_chunk(cues, 2.0)
    assert rendered.splitlines()[0].startswith("[1] (<=8 words) The Secretary")
    assert "\n" not in rendered.splitlines()[0]  # multi-line cue text is flattened


def test_parse_reply_handles_wrapping_and_ignores_strays() -> None:
    reply = (
        "[1] The department decided on the application.\n"
        "extra wrapped words\n"
        "[9] not expected\n"
        "[3] 4,200 households in Harris County are not affected."
    )
    got = parse_reply(reply, {1, 2, 3})
    assert got[1] == "The department decided on the application. extra wrapped words"
    assert 9 not in got
    assert 2 not in got
    assert got[3].startswith("4,200 households")


class FakeClient:
    """Stands in for AnthropicBedrockMantle. Records calls, replays scripted replies."""

    def __init__(self, replies: list[Any]) -> None:
        self.replies = list(replies)
        self.calls: list[dict[str, Any]] = []
        self.messages = self

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


class FakeBlock:
    def __init__(self, text: str) -> None:
        self.type = "text"
        self.text = text


class FakeResponse:
    def __init__(self, text: str, stop_reason: str = "end_turn") -> None:
        self.content = [FakeBlock(text)]
        self.stop_reason = stop_reason


def _simplifier(replies: list[Any], **kwargs: Any) -> BedrockSimplifier:
    return BedrockSimplifier(client=FakeClient(replies), **kwargs)


def test_null_simplifier_produces_nothing() -> None:
    assert NullSimplifier().simplify(VTT) is None
    assert NullSimplifier().name == "none"


def test_simplify_rewrites_text_and_keeps_timings() -> None:
    reply = FakeResponse(
        "[1] The department decided on the application.\n"
        "[2] [APPLAUSE]\n"
        "[3] About 4,200 homes in Harris County are not affected."
    )
    s = _simplifier([reply])
    out = s.simplify(VTT)
    assert out is not None
    cues, _ = parse_vtt(out)
    original, _ = parse_vtt(VTT)
    assert [c.start_s for c in cues] == [c.start_s for c in original]
    assert [c.end_s for c in cues] == [c.end_s for c in original]
    assert cues[0].text == "The department decided on the application."
    assert cues[2].text.startswith("About 4,200 homes")
    assert s.stats.rewritten == 3
    assert s.stats.kept_verbatim == 0
    assert s.stats.coverage == 1.0


def test_missing_cue_is_retried_then_kept_verbatim() -> None:
    first = FakeResponse("[1] Short.\n[3] Also short.")
    retry = FakeResponse("nothing useful here")
    s = _simplifier([first, retry])
    out = s.simplify(VTT)
    assert out is not None
    cues, _ = parse_vtt(out)
    assert cues[1].text == "[APPLAUSE]"  # cue 2 kept its original text
    assert s.stats.retries == 1
    assert s.stats.kept_verbatim == 1
    assert s.stats.rewritten == 2


def test_api_failure_keeps_every_cue_verbatim_and_returns_none() -> None:
    s = _simplifier([RuntimeError("throttled"), RuntimeError("throttled again")])
    assert s.simplify(VTT) is None
    assert s.stats.rewritten == 0
    assert s.stats.failures


def test_refusal_and_max_tokens_are_treated_as_failures() -> None:
    for stop in ("refusal", "max_tokens"):
        s = _simplifier([FakeResponse("[1] x", stop_reason=stop)] * 2)
        assert s.simplify(VTT) is None
        assert any(stop in f or "declined" in f or "max_tokens" in f for f in s.stats.failures)


def test_chunking_splits_requests() -> None:
    replies = [FakeResponse("[1] a\n[2] b"), FakeResponse("[3] c")]
    client = FakeClient(replies)
    s = BedrockSimplifier(client=client, chunk_size=2)
    s.simplify(VTT)
    assert s.stats.chunks == 2
    assert len(client.calls) == 2
    assert client.calls[0]["model"] == "anthropic.claude-opus-5"
    assert client.calls[0]["output_config"] == {"effort": "low"}
    assert "captions" in client.calls[0]["system"].lower()


def test_empty_vtt_returns_none() -> None:
    assert _simplifier([]).simplify("WEBVTT\n") is None


def test_simplifier_name_reports_the_model() -> None:
    s = _simplifier([], model="anthropic.claude-sonnet-5")
    assert s.name == "amazon-bedrock:anthropic.claude-sonnet-5"


def test_both_simplifiers_satisfy_the_protocol() -> None:
    a: Simplifier = NullSimplifier()
    b: Simplifier = _simplifier([])
    assert a.name and b.name


def test_generated_caption_from_simplifier_is_never_reviewed() -> None:
    s = _simplifier([], model="anthropic.claude-sonnet-5")
    cap = Caption.generated(CaptionKind.SIMPLIFIED, "en", "s.vtt", s.name)
    assert cap.reviewed is False
    assert cap.to_dict()["generatedBy"] == "amazon-bedrock:anthropic.claude-sonnet-5"
