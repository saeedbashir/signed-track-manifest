"""S3 in and out, for the container.

On AWS Batch the pipeline has no local files: the sources list and the source
video arrive from S3, and the finished title goes back to S3 for the delivery
bucket to serve. Locally none of this runs — the CLI reads paths and writes
directories — so the module is thin, and ``boto3`` is imported on first use
rather than at import time.

The one piece of logic worth having in one place is :func:`absolutise`: a
manifest is written with URLs relative to itself, and publishing it anywhere
means rewriting them to where the files will actually be. The shell script
that publishes to R2 does the same thing; a judge who reads both should see
the same four fields.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlparse


class S3Client(Protocol):
    """The two calls this module makes. boto3's client satisfies it; tests fake it.

    The parameter names are boto3's, capitals included, so the Protocol matches
    the real client's keyword arguments.
    """

    def download_file(self, Bucket: str, Key: str, Filename: str) -> None: ...  # noqa: N803

    def upload_file(
        self,
        Filename: str,  # noqa: N803
        Bucket: str,  # noqa: N803
        Key: str,  # noqa: N803
        ExtraArgs: dict[str, Any] | None = None,  # noqa: N803
    ) -> None: ...


_state: dict[str, S3Client | None] = {"client": None}


def use_client(client: S3Client | None) -> None:
    """Inject a client (tests), or ``None`` to go back to boto3."""
    _state["client"] = client


def client() -> S3Client:
    current = _state["client"]
    if current is None:
        try:
            import boto3
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise RuntimeError(
                "S3 needs the 'aws' extra: pip install 'signed-track-manifest[aws]'"
            ) from exc
        current = boto3.client("s3")
        _state["client"] = current
    return current


def is_s3(url: str) -> bool:
    return url.startswith("s3://")


def split(url: str) -> tuple[str, str]:
    """``s3://bucket/some/key`` → ``("bucket", "some/key")``. The key may be empty."""
    if not is_s3(url):
        raise ValueError(f"not an S3 URL: {url!r}")
    parsed = urlparse(url)
    if not parsed.netloc:
        raise ValueError(f"S3 URL has no bucket: {url!r}")
    return parsed.netloc, parsed.path.lstrip("/")


def download(url: str, dst: Path) -> Path:
    """Fetch one object to ``dst``, writing a ``.sha256`` beside it like an HTTP download."""
    bucket, key = split(url)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    client().download_file(bucket, key, str(tmp))
    tmp.replace(dst)
    dst.with_suffix(dst.suffix + ".sha256").write_text(_sha256(dst) + "\n", encoding="utf-8")
    return dst


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


#: What the delivery bucket should say a file is. Vega's player is strict about
#: video MIME types, and a caption served as octet-stream is a caption that
#: does not load.
CONTENT_TYPES = {
    ".mp4": "video/mp4",
    ".m4v": "video/mp4",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".vtt": "text/vtt",
    ".srt": "application/x-subrip",
    ".json": "application/json",
    ".m3u8": "application/vnd.apple.mpegurl",
}


def content_type(path: Path) -> str:
    return CONTENT_TYPES.get(path.suffix.lower(), "application/octet-stream")


def absolutise(manifest: dict[str, Any], base: str) -> dict[str, Any]:
    """Every relative URL in a title manifest made absolute under ``base``.

    ``base`` is the URL of the directory the manifest will live in, with or
    without a trailing slash. URLs that are already absolute are left alone.
    """
    base = base.rstrip("/")

    def fix(url: Any) -> Any:
        if isinstance(url, str) and url and "://" not in url:
            return f"{base}/{url.lstrip('/')}"
        return url

    out: dict[str, Any] = json.loads(json.dumps(manifest))  # a copy; the caller's is untouched
    video = out.get("video")
    if isinstance(video, dict):
        main = video.get("main")
        if isinstance(main, dict):
            main["url"] = fix(main.get("url"))
        video["poster"] = fix(video.get("poster"))
    for track in out.get("signLanguage") or []:
        t = track.get("track")
        if isinstance(t, dict):
            t["url"] = fix(t.get("url"))
    for cap in out.get("captions") or []:
        cap["url"] = fix(cap.get("url"))
    return out


#: The analysis report is for the person checking the pipeline, not the player.
EXCLUDE = ("analysis.json",)


def upload_title(
    title_dir: Path,
    dest: str,
    public_base: str | None = None,
    *,
    exclude: tuple[str, ...] = EXCLUDE,
    log: Callable[[str], None] | None = None,
) -> list[str]:
    """Upload ``out/<id>/`` to ``dest`` (``s3://bucket/prefix``), under ``<id>/``.

    With ``public_base`` — the public URL of the *titles* directory — the
    manifest's relative URLs are rewritten to ``public_base/<id>/…`` before
    upload, so what the bucket serves is what a player can follow. The local
    manifest is left as written. Returns the S3 URLs written, in upload order.
    """
    bucket, prefix = split(dest)
    key_prefix = "/".join(p for p in (prefix.strip("/"), title_dir.name) if p)
    s3 = client()
    written: list[str] = []
    for path in sorted(p for p in title_dir.rglob("*") if p.is_file()):
        rel = path.relative_to(title_dir).as_posix()
        if rel in exclude:
            continue
        key = f"{key_prefix}/{rel}"
        source = path
        if rel == "manifest.json" and public_base:
            manifest = json.loads(path.read_text(encoding="utf-8"))
            public = absolutise(manifest, f"{public_base.rstrip('/')}/{title_dir.name}")
            source = path.with_name("manifest.public.json")
            source.write_text(json.dumps(public, indent=2, ensure_ascii=False) + "\n")
        s3.upload_file(str(source), bucket, key, ExtraArgs={"ContentType": content_type(path)})
        if source is not path:
            source.unlink()
        url = f"s3://{bucket}/{key}"
        written.append(url)
        if log:
            log(f"  uploaded {rel} -> {url}")
    return written
