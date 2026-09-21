"""Typed model of a Signed Track Manifest.

The dataclasses here mirror the JSON Schema and add the two rules the schema
alone cannot express as intent:

* ``provenance`` is required and has no default. A human-interpreter track must
  carry ``extraction``; a synthesised track must carry ``generatedBy``.
* A machine-generated caption is always written with ``reviewed: false``. The
  only way to mark it reviewed is :meth:`Caption.mark_reviewed`, which requires
  a named reviewer. No configuration flag can do it.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Any

STM_VERSION = "0.1"

#: Below this extraction confidence a player should badge the title.
LOW_CONFIDENCE_THRESHOLD = 0.7


class Provenance(StrEnum):
    HUMAN_INTERPRETER = "human-interpreter"
    SYNTHESISED = "synthesised"


class SourceLayout(StrEnum):
    CORNER_INSET = "corner-inset"
    SIDE_PANEL = "side-panel"


class ExtractionMethod(StrEnum):
    DETECTED_CROP = "detected-crop"
    MANUAL_CROP = "manual-crop"


class FillMethod(StrEnum):
    NONE = "none"
    INTERPOLATED_SURROUND = "interpolated-surround"
    BLURRED_SURROUND = "blurred-surround"


class CaptionKind(StrEnum):
    VERBATIM = "verbatim"
    SIMPLIFIED = "simplified"


def _drop_none(d: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in d.items() if v is not None}


@dataclass(frozen=True)
class Rect:
    """An axis-aligned rectangle in integer pixels, origin top-left."""

    x: int
    y: int
    w: int
    h: int

    def __post_init__(self) -> None:
        if self.w <= 0 or self.h <= 0:
            raise ValueError(f"Rect needs positive width and height, got {self.w}x{self.h}")
        if self.x < 0 or self.y < 0:
            raise ValueError(f"Rect origin must be non-negative, got ({self.x}, {self.y})")

    @property
    def x2(self) -> int:
        return self.x + self.w

    @property
    def y2(self) -> int:
        return self.y + self.h

    @property
    def area(self) -> int:
        return self.w * self.h

    def iou(self, other: Rect) -> float:
        """Intersection over union with another rectangle."""
        ix = max(0, min(self.x2, other.x2) - max(self.x, other.x))
        iy = max(0, min(self.y2, other.y2) - max(self.y, other.y))
        inter = ix * iy
        union = self.area + other.area - inter
        return inter / union if union else 0.0

    def contains(self, other: Rect) -> bool:
        return (
            self.x <= other.x and self.y <= other.y and self.x2 >= other.x2 and self.y2 >= other.y2
        )

    def to_dict(self) -> dict[str, int]:
        return {"x": self.x, "y": self.y, "w": self.w, "h": self.h}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Rect:
        return cls(int(d["x"]), int(d["y"]), int(d["w"]), int(d["h"]))


@dataclass(frozen=True)
class Source:
    publisher: str
    url: str
    licence: str
    attribution: str
    production_credit: str | None = None
    acquisition_route: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return _drop_none(
            {
                "publisher": self.publisher,
                "url": self.url,
                "licence": self.licence,
                "attribution": self.attribution,
                "productionCredit": self.production_credit,
                "acquisitionRoute": self.acquisition_route,
            }
        )


@dataclass(frozen=True)
class VideoAsset:
    url: str
    width: int
    height: int
    codec: str = "h264"

    def to_dict(self) -> dict[str, Any]:
        return {"url": self.url, "width": self.width, "height": self.height, "codec": self.codec}


@dataclass(frozen=True)
class Extraction:
    method: ExtractionMethod
    source_layout: SourceLayout
    confidence: float
    crop_rect: Rect
    fill_method: FillMethod

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be within 0..1, got {self.confidence}")

    @property
    def low_confidence(self) -> bool:
        return self.confidence < LOW_CONFIDENCE_THRESHOLD

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": str(self.method),
            "sourceLayout": str(self.source_layout),
            "confidence": round(self.confidence, 4),
            "cropRect": self.crop_rect.to_dict(),
            "fillMethod": str(self.fill_method),
        }


@dataclass(frozen=True)
class SignTrack:
    """One sign language track. ``provenance`` has no default on purpose."""

    language: str
    language_label: str
    provenance: Provenance
    track: VideoAsset
    sync_offset_ms: int = 0
    extraction: Extraction | None = None
    generated_by: str | None = None

    def __post_init__(self) -> None:
        if len(self.language) != 3 or not self.language.isalpha() or not self.language.islower():
            raise ValueError(
                f"language must be an ISO 639-3 code such as 'ase', got {self.language!r}"
            )
        if self.provenance is Provenance.HUMAN_INTERPRETER:
            if self.extraction is None:
                raise ValueError("a human-interpreter track must record how it was extracted")
            if self.generated_by is not None:
                raise ValueError("a human-interpreter track cannot carry generatedBy")
        else:
            if not self.generated_by:
                raise ValueError("a synthesised track must name what generated it")
            if self.extraction is not None:
                raise ValueError("a synthesised track has no extraction")

    def to_dict(self) -> dict[str, Any]:
        return _drop_none(
            {
                "language": self.language,
                "languageLabel": self.language_label,
                "provenance": str(self.provenance),
                "extraction": self.extraction.to_dict() if self.extraction else None,
                "generatedBy": self.generated_by,
                "track": {
                    **self.track.to_dict(),
                    "syncOffsetMs": self.sync_offset_ms,
                },
            }
        )


@dataclass(frozen=True)
class Caption:
    """A caption track.

    Construct machine-generated tracks with :meth:`generated`; they are always
    ``reviewed=False``. Marking one reviewed requires a named reviewer.
    """

    kind: CaptionKind
    language: str
    url: str
    format: str = "webvtt"
    generated_by: str | None = None
    reviewed: bool | None = None
    reviewed_by: str | None = None

    def __post_init__(self) -> None:
        if self.generated_by is not None and self.reviewed is None:
            object.__setattr__(self, "reviewed", False)
        if self.reviewed is True and not (self.reviewed_by and self.reviewed_by.strip()):
            raise ValueError("reviewed=True requires reviewed_by to name the reviewer")
        if self.reviewed is not True and self.reviewed_by is not None:
            raise ValueError("reviewed_by is only meaningful when reviewed is True")

    @classmethod
    def generated(
        cls,
        kind: CaptionKind,
        language: str,
        url: str,
        generated_by: str,
        fmt: str = "webvtt",
    ) -> Caption:
        """The only constructor the pipeline uses for machine output. Never reviewed."""
        if not generated_by.strip():
            raise ValueError("generated_by must name the generator")
        return cls(
            kind=kind,
            language=language,
            url=url,
            format=fmt,
            generated_by=generated_by.strip(),
            reviewed=False,
        )

    def mark_reviewed(self, reviewer: str) -> Caption:
        """Return a copy marked as human-reviewed by a named person."""
        if not reviewer.strip():
            raise ValueError("a reviewer name is required to mark a caption reviewed")
        return replace(self, reviewed=True, reviewed_by=reviewer.strip())

    def to_dict(self) -> dict[str, Any]:
        return _drop_none(
            {
                "kind": str(self.kind),
                "language": self.language,
                "url": self.url,
                "format": self.format,
                "generatedBy": self.generated_by,
                "reviewed": self.reviewed,
                "reviewedBy": self.reviewed_by,
            }
        )


@dataclass(frozen=True)
class TitleManifest:
    id: str
    title: str
    duration_ms: int
    source: Source
    main: VideoAsset
    poster: str
    sign_language: Sequence[SignTrack] = ()
    captions: Sequence[Caption] = ()
    stm_version: str = STM_VERSION

    @property
    def min_confidence(self) -> float | None:
        values = [t.extraction.confidence for t in self.sign_language if t.extraction]
        return min(values) if values else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "stmVersion": self.stm_version,
            "id": self.id,
            "title": self.title,
            "durationMs": self.duration_ms,
            "source": self.source.to_dict(),
            "video": {"main": self.main.to_dict(), "poster": self.poster},
            "signLanguage": [t.to_dict() for t in self.sign_language],
            "captions": [c.to_dict() for c in self.captions],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False) + "\n"


@dataclass(frozen=True)
class CatalogueEntry:
    id: str
    title: str
    duration_ms: int
    poster: str
    manifest_url: str
    sign_languages: Sequence[str]
    confidence: float | None = None

    @classmethod
    def from_manifest(cls, manifest: TitleManifest, manifest_url: str) -> CatalogueEntry:
        return cls(
            id=manifest.id,
            title=manifest.title,
            duration_ms=manifest.duration_ms,
            poster=manifest.poster,
            manifest_url=manifest_url,
            sign_languages=[t.language for t in manifest.sign_language],
            confidence=manifest.min_confidence,
        )

    def to_dict(self) -> dict[str, Any]:
        return _drop_none(
            {
                "id": self.id,
                "title": self.title,
                "durationMs": self.duration_ms,
                "poster": self.poster,
                "manifestUrl": self.manifest_url,
                "signLanguages": list(self.sign_languages),
                "confidence": None if self.confidence is None else round(self.confidence, 4),
            }
        )


@dataclass(frozen=True)
class Catalogue:
    titles: Sequence[CatalogueEntry]
    stm_version: str = STM_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {"stmVersion": self.stm_version, "titles": [t.to_dict() for t in self.titles]}

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False) + "\n"
