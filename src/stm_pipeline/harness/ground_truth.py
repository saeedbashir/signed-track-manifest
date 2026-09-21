from __future__ import annotations

import contextlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from stm.model import Rect


@dataclass(frozen=True)
class GroundTruth:
    """A hand-drawn rectangle for one clip, in source pixels.

    ``face`` is optional: the rectangle the interpreter's face occupies at its
    most expressive, so the harness can report whether a crop that scores well
    still clips the face. Facial expression carries grammar in signed languages.
    """

    clip: Path
    rect: Rect
    face: Rect | None = None
    notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"clip": str(self.clip), "rect": self.rect.to_dict()}
        if self.face is not None:
            d["face"] = self.face.to_dict()
        if self.notes:
            d["notes"] = self.notes
        return d


def _parse(item: Mapping[str, Any], base: Path) -> GroundTruth:
    clip = Path(str(item["clip"]))
    if not clip.is_absolute():
        clip = base / clip
    face = item.get("face")
    return GroundTruth(
        clip=clip,
        rect=Rect.from_dict(item["rect"]),
        face=Rect.from_dict(face) if isinstance(face, Mapping) else None,
        notes=str(item["notes"]) if item.get("notes") else None,
    )


def load_ground_truth(path: Path) -> list[GroundTruth]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    clips = data.get("clips") if isinstance(data, Mapping) else None
    if clips is None:
        return []
    if not isinstance(clips, list):
        raise ValueError(f"{path}: 'clips' must be a list")
    return [_parse(item, path.parent) for item in clips]


def save_ground_truth(path: Path, items: list[GroundTruth]) -> None:
    base = path.parent
    out = []
    for gt in items:
        d = gt.to_dict()
        with contextlib.suppress(ValueError):
            d["clip"] = str(gt.clip.relative_to(base))
        out.append(d)
    path.write_text(yaml.safe_dump({"clips": out}, sort_keys=False), encoding="utf-8")
