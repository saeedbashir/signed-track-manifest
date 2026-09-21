"""Read ``sources.yaml`` and fetch media without breaking anyone's terms.

Every title records its licence, attribution, production credit and how the
file was obtained before anything is downloaded. Platform rips are refused:
YouTube's terms prohibit downloading except through features it provides,
regardless of the video's copyright status.
"""

from __future__ import annotations

import hashlib
import re
import urllib.parse
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from stm.model import Rect, SourceLayout

BLOCKED_HOSTS = ("youtube.com", "youtu.be", "googlevideo.com")
_ID = re.compile(r"^[a-z0-9][a-z0-9-]{1,63}$")


class SourceError(ValueError):
    """The sources file is malformed or an entry cannot be used."""


class BlockedSourceError(SourceError):
    """The URL points at a platform whose terms forbid downloading."""


@dataclass(frozen=True)
class SourceEntry:
    id: str
    title: str
    url: str
    publisher: str
    licence: str
    attribution: str
    production_credit: str | None = None
    acquisition_route: str | None = None
    local_path: Path | None = None
    captions: str | None = None  # URL or local path to .vtt / .srt
    caption_language: str = "en"
    sign_language: str = "ase"
    language_label: str = "American Sign Language"
    crop_rect: Rect | None = None  # manual override, source pixels
    layout: SourceLayout | None = None  # required with crop_rect
    notes: str | None = None


def _req(d: Mapping[str, Any], key: str, entry_id: str) -> str:
    v = d.get(key)
    if not isinstance(v, str) or not v.strip():
        raise SourceError(f"source {entry_id!r}: {key!r} is required")
    return v.strip()


def _opt(d: Mapping[str, Any], key: str) -> str | None:
    v = d.get(key)
    return v.strip() if isinstance(v, str) and v.strip() else None


def parse_entry(d: Mapping[str, Any]) -> SourceEntry:
    entry_id = str(d.get("id", "?"))
    if not _ID.match(entry_id):
        raise SourceError(f"source id {entry_id!r} must match {_ID.pattern}")
    crop = d.get("crop_rect")
    layout_raw = _opt(d, "layout")
    layout = SourceLayout(layout_raw) if layout_raw else None
    crop_rect = Rect.from_dict(crop) if isinstance(crop, Mapping) else None
    if crop_rect is not None and layout is None:
        raise SourceError(f"source {entry_id!r}: a manual crop_rect needs an explicit layout")
    local = _opt(d, "local_path")
    return SourceEntry(
        id=entry_id,
        title=_req(d, "title", entry_id),
        url=_req(d, "url", entry_id),
        publisher=_req(d, "publisher", entry_id),
        licence=_req(d, "licence", entry_id),
        attribution=_req(d, "attribution", entry_id),
        production_credit=_opt(d, "production_credit"),
        acquisition_route=_opt(d, "acquisition_route"),
        local_path=Path(local) if local else None,
        captions=_opt(d, "captions"),
        caption_language=_opt(d, "caption_language") or "en",
        sign_language=_opt(d, "sign_language") or "ase",
        language_label=_opt(d, "language_label") or "American Sign Language",
        crop_rect=crop_rect,
        layout=layout,
        notes=_opt(d, "notes"),
    )


def load_sources(path: Path) -> list[SourceEntry]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    raw = data.get("sources") if isinstance(data, Mapping) else None
    if not isinstance(raw, list):
        raise SourceError(f"{path}: expected a top-level 'sources' list")
    entries = [parse_entry(item) for item in raw]
    ids = [e.id for e in entries]
    dupes = {i for i in ids if ids.count(i) > 1}
    if dupes:
        raise SourceError(f"{path}: duplicate source ids {sorted(dupes)}")
    return entries


def check_url_allowed(url: str) -> None:
    host = (urllib.parse.urlparse(url).hostname or "").lower()
    if any(host == b or host.endswith("." + b) for b in BLOCKED_HOSTS):
        raise BlockedSourceError(
            f"{host} is a platform whose terms forbid downloading. Obtain the file from the "
            "publisher, an agency download page or a public archive, and record the route "
            "in acquisition_route."
        )


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, dst: Path) -> None:
    check_url_allowed(url)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "signed-track-manifest/0.1"})
    with urllib.request.urlopen(req) as resp, tmp.open("wb") as out:
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            out.write(chunk)
    tmp.replace(dst)
    dst.with_suffix(dst.suffix + ".sha256").write_text(_sha256(dst) + "\n", encoding="utf-8")


def fetch_media(entry: SourceEntry, work_dir: Path) -> Path:
    """Return a local path to the source video, downloading if needed."""
    if entry.local_path is not None:
        if not entry.local_path.exists():
            raise SourceError(f"source {entry.id!r}: local_path {entry.local_path} does not exist")
        return entry.local_path
    suffix = Path(urllib.parse.urlparse(entry.url).path).suffix or ".mp4"
    dst = work_dir / entry.id / f"source{suffix}"
    if not dst.exists():
        _download(entry.url, dst)
    return dst


def fetch_captions(entry: SourceEntry, work_dir: Path) -> Path | None:
    """Return a local path to the source captions, downloading if given as a URL."""
    if not entry.captions:
        return None
    if "://" not in entry.captions:
        p = Path(entry.captions)
        if not p.exists():
            raise SourceError(f"source {entry.id!r}: captions file {p} does not exist")
        return p
    suffix = Path(urllib.parse.urlparse(entry.captions).path).suffix or ".vtt"
    dst = work_dir / entry.id / f"captions{suffix}"
    if not dst.exists():
        _download(entry.captions, dst)
    return dst
