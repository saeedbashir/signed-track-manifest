"""Speech recognition: words with times, from the programme's audio.

Two uses, and the distinction is the point of the module.

The first is as a **clock** for a transcript a human wrote — the aligner takes
these words only for their timestamps and publishes the human text. The second
is as a transcript in its own right where no written record exists; that track
is machine text and is published as such: ``generatedBy`` set, ``reviewed:
false``, badged by the player.

Behind the optional ``captions`` extra (``faster-whisper``, a CTranslate2
build of OpenAI's Whisper): CPU only, no torch, about five times real time on a
laptop with the ``small`` model. Importing this module costs nothing; the model
is loaded on first use, and a missing extra names itself.
"""

from __future__ import annotations

import hashlib
import json
import time
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

from stm_pipeline import ffmpeg

DEFAULT_MODEL = "small"
KNOWN_MODELS = ("tiny", "base", "small", "medium", "large-v3")


@dataclass(frozen=True)
class TimedWord:
    """One recognised word and when it was said, in seconds from the start."""

    text: str
    start: float
    end: float
    probability: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TimedWord:
        return cls(
            text=str(d["text"]),
            start=float(d["start"]),
            end=float(d["end"]),
            probability=float(d.get("probability", 1.0)),
        )


def generated_by(model: str) -> str:
    """The ``generatedBy`` value for a track whose *text* this module wrote."""
    return f"faster-whisper-{model}"


def extract_audio(video: Path, wav: Path) -> Path:
    """16 kHz mono PCM, which is what the recogniser wants and what keeps decoding cheap."""
    wav.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg.run_ffmpeg(["-i", str(video), "-vn", "-ac", "1", "-ar", "16000", str(wav)])
    return wav


def transcribe(audio: Path, model: str = DEFAULT_MODEL, language: str = "en") -> list[TimedWord]:
    """Every word in the audio with its start and end, in order."""
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise RuntimeError(
            "speech recognition needs the 'captions' extra: "
            "pip install 'signed-track-manifest[captions]'"
        ) from exc
    engine = WhisperModel(model, device="cpu", compute_type="int8")
    segments, _info = engine.transcribe(
        str(audio), language=language, word_timestamps=True, vad_filter=True
    )
    words: list[TimedWord] = []
    for segment in segments:
        for w in segment.words or []:
            text = w.word.strip()
            if text:
                words.append(TimedWord(text, float(w.start), float(w.end), float(w.probability)))
    return words


def save_words(words: list[TimedWord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([w.to_dict() for w in words], ensure_ascii=False), encoding="utf-8")


def load_words(path: Path) -> list[TimedWord]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [TimedWord.from_dict(d) for d in data]


def transcribe_cached(
    video: Path, work_dir: Path, model: str = DEFAULT_MODEL, language: str = "en"
) -> list[TimedWord]:
    """Transcribe once per video and model; a second run reads the JSON instead.

    A 34-minute session costs six minutes of CPU. Aligning it against a
    corrected transcript, or re-cutting cues, should not cost that again.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    cache = work_dir / f"asr.{model}.{language}.json"
    if cache.exists():
        return load_words(cache)
    wav = work_dir / "audio.16k.wav"
    if not wav.exists():
        extract_audio(video, wav)
    words = transcribe(wav, model, language)
    save_words(words, cache)
    return words


# --- Amazon Transcribe ---------------------------------------------------------
#
# A second clock, for the AWS Batch path. Measured on the 32-minute Backbench
# session against the Official Report: Whisper `small` anchors 85.3 % of the
# Report's words, Transcribe 83.8 %, cue for cue within a few hundredths of a
# second — so Whisper stays the default, and Transcribe is chosen where it is
# the better fit: on AWS, where it is a managed service, needs no model download
# and takes 80 seconds for that session to Whisper's six minutes on a laptop.
# Its text, like Whisper's, is used only for the timings.

#: The recognisers `stm` knows. ``whisper`` runs locally; ``transcribe`` calls AWS.
RECOGNISERS = ("whisper", "transcribe")
DEFAULT_TRANSCRIBE_LANGUAGE = "en-GB"


class TranscribeClient(Protocol):
    """The two Transcribe calls used here. boto3's client satisfies it; tests fake it."""

    def start_transcription_job(self, **kwargs: Any) -> dict[str, Any]: ...

    def get_transcription_job(self, **kwargs: Any) -> dict[str, Any]: ...


def parse_transcribe(result: dict[str, Any]) -> list[TimedWord]:
    """Words with times from a Transcribe result document.

    Punctuation arrives as its own item with no time; it is attached to the word
    before it, as Whisper writes it, so the aligner sees the same shapes.
    """
    words: list[TimedWord] = []
    for item in result.get("results", {}).get("items", []):
        alt = (item.get("alternatives") or [{}])[0]
        text = str(alt.get("content", "")).strip()
        if not text:
            continue
        if item.get("type") == "pronunciation":
            words.append(
                TimedWord(
                    text,
                    float(item["start_time"]),
                    float(item["end_time"]),
                    float(alt.get("confidence", 1.0)),
                )
            )
        elif words:
            last = words[-1]
            words[-1] = TimedWord(last.text + text, last.start, last.end, last.probability)
    return words


def _job_name(media: str) -> str:
    digest = hashlib.sha256(media.encode("utf-8")).hexdigest()[:12]
    return f"stm-{digest}-{int(time.time())}"


def transcribe_aws(
    media_uri: str,
    language_code: str = DEFAULT_TRANSCRIBE_LANGUAGE,
    *,
    client: TranscribeClient | None = None,
    poll_seconds: float = 10.0,
    timeout_seconds: float = 3600.0,
    fetch: Any = None,
) -> list[TimedWord]:
    """Transcribe an ``s3://`` media object and return its words with times.

    No output bucket is given, so Transcribe keeps the result and returns a
    short-lived link to it: the caller needs read access to the media and
    nothing else.
    """
    if not media_uri.startswith("s3://"):
        raise ValueError(f"Transcribe reads from S3; got {media_uri!r}")
    if client is None:
        try:
            import boto3
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise RuntimeError(
                "Amazon Transcribe needs the 'aws' extra: pip install 'signed-track-manifest[aws]'"
            ) from exc
        client = boto3.client("transcribe")
    name = _job_name(media_uri)
    client.start_transcription_job(
        TranscriptionJobName=name,
        LanguageCode=language_code,
        Media={"MediaFileUri": media_uri},
    )
    deadline = time.monotonic() + timeout_seconds
    while True:
        job = client.get_transcription_job(TranscriptionJobName=name)["TranscriptionJob"]
        status = job["TranscriptionJobStatus"]
        if status == "COMPLETED":
            break
        if status == "FAILED":
            raise RuntimeError(f"Transcribe job {name} failed: {job.get('FailureReason')}")
        if time.monotonic() > deadline:
            raise RuntimeError(
                f"Transcribe job {name} still {status} after {timeout_seconds:.0f} s"
            )
        time.sleep(poll_seconds)
    uri = job["Transcript"]["TranscriptFileUri"]
    if fetch is None:
        with urllib.request.urlopen(uri, timeout=120) as response:
            document = json.loads(response.read().decode("utf-8"))
    else:
        document = fetch(uri)
    return parse_transcribe(document)


def transcribe_aws_cached(
    media_uri: str, work_dir: Path, language_code: str = DEFAULT_TRANSCRIBE_LANGUAGE, **kwargs: Any
) -> list[TimedWord]:
    """As :func:`transcribe_cached`, for Transcribe: once per media and language."""
    work_dir.mkdir(parents=True, exist_ok=True)
    cache = work_dir / f"asr.transcribe.{language_code}.json"
    if cache.exists():
        return load_words(cache)
    words = transcribe_aws(media_uri, language_code, **kwargs)
    save_words(words, cache)
    return words
