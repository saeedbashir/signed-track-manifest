"""Amazon Transcribe as the clock: result parsing and the job loop, against fakes."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from stm_pipeline.asr import load_words, parse_transcribe, transcribe_aws, transcribe_aws_cached

RESULT: dict[str, Any] = {
    "results": {
        "items": [
            {"type": "pronunciation", "start_time": "7.62", "end_time": "8.01",
             "alternatives": [{"confidence": "0.98", "content": "Last"}]},
            {"type": "pronunciation", "start_time": "8.01", "end_time": "8.40",
             "alternatives": [{"confidence": "0.97", "content": "week"}]},
            {"type": "punctuation", "alternatives": [{"confidence": "0.0", "content": ","}]},
            {"type": "pronunciation", "start_time": "8.50", "end_time": "8.70",
             "alternatives": [{"confidence": "0.9", "content": "the"}]},
            {"type": "punctuation", "alternatives": [{"confidence": "0.0", "content": "."}]},
        ]
    }
}  # fmt: skip


class FakeTranscribe:
    def __init__(self, statuses: list[str]) -> None:
        self.statuses = statuses
        self.started: list[dict[str, Any]] = []

    def start_transcription_job(self, **kwargs: Any) -> dict[str, Any]:
        self.started.append(kwargs)
        return {}

    def get_transcription_job(self, **kwargs: Any) -> dict[str, Any]:
        status = self.statuses.pop(0)
        job: dict[str, Any] = {"TranscriptionJobStatus": status, "FailureReason": "bad audio"}
        if status == "COMPLETED":
            job["Transcript"] = {"TranscriptFileUri": "https://example/result.json"}
        return {"TranscriptionJob": job}


def test_punctuation_attaches_to_the_word_before_it() -> None:
    words = parse_transcribe(RESULT)
    assert [w.text for w in words] == ["Last", "week,", "the."]
    assert words[1].start == 8.01 and words[1].end == 8.40
    assert words[0].probability == pytest.approx(0.98)


def test_job_polls_until_complete_and_needs_no_output_bucket() -> None:
    fake = FakeTranscribe(["IN_PROGRESS", "IN_PROGRESS", "COMPLETED"])
    words = transcribe_aws(
        "s3://src/a.mp4", "en-GB", client=fake, poll_seconds=0, fetch=lambda uri: RESULT
    )
    assert len(words) == 3
    (call,) = fake.started
    assert call["Media"] == {"MediaFileUri": "s3://src/a.mp4"}
    assert call["LanguageCode"] == "en-GB"
    assert "OutputBucketName" not in call


def test_failed_job_raises_with_the_reason() -> None:
    with pytest.raises(RuntimeError, match="bad audio"):
        transcribe_aws("s3://src/a.mp4", client=FakeTranscribe(["FAILED"]), poll_seconds=0)


def test_local_paths_are_refused() -> None:
    with pytest.raises(ValueError):
        transcribe_aws("/tmp/a.mp4", client=FakeTranscribe([]))


def test_cached_result_is_reused(tmp_path: Path) -> None:
    fake = FakeTranscribe(["COMPLETED"])
    first = transcribe_aws_cached(
        "s3://src/a.mp4", tmp_path, client=fake, poll_seconds=0, fetch=lambda uri: RESULT
    )
    again = transcribe_aws_cached("s3://src/a.mp4", tmp_path, client=FakeTranscribe([]))
    assert first == again == load_words(tmp_path / "asr.transcribe.en-GB.json")
