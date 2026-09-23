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

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

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
