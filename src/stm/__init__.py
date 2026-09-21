"""Signed Track Manifest (STM): the format, a typed model, and a validator.

STM describes a title whose sign language interpretation is delivered as a
separate video layer, so a player can position, resize, or hide it. It records
where each signing track came from (`provenance`), how it was extracted, and
how confident the extraction was.

This package never generates or synthesises sign language. It describes and
validates manifests.
"""

from stm.model import (
    LOW_CONFIDENCE_THRESHOLD,
    STM_VERSION,
    Caption,
    CaptionKind,
    Catalogue,
    CatalogueEntry,
    Extraction,
    ExtractionMethod,
    FillMethod,
    Provenance,
    Rect,
    SignTrack,
    Source,
    SourceLayout,
    TitleManifest,
    VideoAsset,
)
from stm.schema import Issue, is_valid, validate_catalogue, validate_title

__version__ = "0.1.0"

__all__ = [
    "LOW_CONFIDENCE_THRESHOLD",
    "STM_VERSION",
    "Caption",
    "CaptionKind",
    "Catalogue",
    "CatalogueEntry",
    "Extraction",
    "ExtractionMethod",
    "FillMethod",
    "Issue",
    "Provenance",
    "Rect",
    "SignTrack",
    "Source",
    "SourceLayout",
    "TitleManifest",
    "VideoAsset",
    "is_valid",
    "validate_catalogue",
    "validate_title",
]
