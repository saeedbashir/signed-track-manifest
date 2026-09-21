"""JSON Schema validation for STM manifests, plus a few semantic checks."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from functools import cache
from importlib import resources
from typing import Any, Literal

from jsonschema import Draft202012Validator

from stm.model import LOW_CONFIDENCE_THRESHOLD

Severity = Literal["error", "warning"]

#: Sync offsets beyond this are almost certainly a mistake, not a lag.
_SYNC_OFFSET_WARN_MS = 5000


@dataclass(frozen=True)
class Issue:
    """One validation finding. ``path`` is a JSON-pointer-like path, ``$`` for the root."""

    path: str
    message: str
    severity: Severity = "error"

    def __str__(self) -> str:
        return f"{self.path}: [{self.severity}] {self.message}"


@cache
def _load_schema(name: str) -> dict[str, Any]:
    text = resources.files("stm.schemas").joinpath(name).read_text(encoding="utf-8")
    schema: dict[str, Any] = json.loads(text)
    return schema


def title_schema() -> dict[str, Any]:
    return _load_schema("stm-title.schema.json")


def catalogue_schema() -> dict[str, Any]:
    return _load_schema("stm-catalogue.schema.json")


def _schema_issues(schema: Mapping[str, Any], data: Any) -> list[Issue]:
    validator = Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(data), key=lambda e: list(map(str, e.absolute_path)))
    issues: list[Issue] = []
    for err in errors:
        path = "/".join(str(p) for p in err.absolute_path) or "$"
        issues.append(Issue(path=path, message=err.message))
    return issues


def _title_semantics(data: Mapping[str, Any]) -> list[Issue]:
    issues: list[Issue] = []

    seen_langs: set[str] = set()
    for i, track in enumerate(data.get("signLanguage", [])):
        lang = track.get("language")
        if lang in seen_langs:
            issues.append(
                Issue(f"signLanguage/{i}/language", f"duplicate sign language track for {lang!r}")
            )
        seen_langs.add(lang)

        extraction = track.get("extraction")
        if extraction is not None:
            conf = extraction.get("confidence", 1.0)
            if conf < LOW_CONFIDENCE_THRESHOLD:
                issues.append(
                    Issue(
                        f"signLanguage/{i}/extraction/confidence",
                        f"extraction confidence {conf} is below {LOW_CONFIDENCE_THRESHOLD}; "
                        "players should badge this title",
                        severity="warning",
                    )
                )
        offset = track.get("track", {}).get("syncOffsetMs", 0)
        if abs(offset) > _SYNC_OFFSET_WARN_MS:
            issues.append(
                Issue(
                    f"signLanguage/{i}/track/syncOffsetMs",
                    f"sync offset of {offset} ms is unusually large",
                    severity="warning",
                )
            )

    seen_caps: set[tuple[str, str]] = set()
    for i, cap in enumerate(data.get("captions", [])):
        key = (cap.get("kind", ""), cap.get("language", ""))
        if key in seen_caps:
            issues.append(
                Issue(f"captions/{i}", f"duplicate {key[0]} caption track for {key[1]!r}")
            )
        seen_caps.add(key)

    return issues


def _catalogue_semantics(data: Mapping[str, Any]) -> list[Issue]:
    issues: list[Issue] = []
    seen: set[str] = set()
    for i, entry in enumerate(data.get("titles", [])):
        tid = entry.get("id")
        if tid in seen:
            issues.append(Issue(f"titles/{i}/id", f"duplicate title id {tid!r}"))
        seen.add(tid)
    return issues


def validate_title(data: Any) -> list[Issue]:
    """Validate a per-title manifest. Returns issues; empty means valid."""
    issues = _schema_issues(title_schema(), data)
    if not issues and isinstance(data, Mapping):
        issues.extend(_title_semantics(data))
    return issues


def validate_catalogue(data: Any) -> list[Issue]:
    """Validate a catalogue index. Returns issues; empty means valid."""
    issues = _schema_issues(catalogue_schema(), data)
    if not issues and isinstance(data, Mapping):
        issues.extend(_catalogue_semantics(data))
    return issues


def is_valid(issues: Iterable[Issue], warnings_as_errors: bool = False) -> bool:
    """True when no issue is an error (or, with the flag, when there are none at all)."""
    return not any(i.severity == "error" or warnings_as_errors for i in issues)
