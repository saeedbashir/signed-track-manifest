"""S3 in and out, against a fake client: nothing here touches the network."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from stm_pipeline import ingest, publish


class FakeS3:
    """Records uploads; serves downloads from a dict of bucket/key -> bytes."""

    def __init__(self, objects: dict[str, bytes] | None = None) -> None:
        self.objects = objects or {}
        self.uploads: list[tuple[str, str, str, dict[str, Any] | None]] = []
        self.snapshots: dict[str, bytes] = {}

    def download_file(self, Bucket: str, Key: str, Filename: str) -> None:  # noqa: N803
        Path(Filename).write_bytes(self.objects[f"{Bucket}/{Key}"])

    def upload_file(
        self,
        Filename: str,  # noqa: N803
        Bucket: str,  # noqa: N803
        Key: str,  # noqa: N803
        ExtraArgs: dict[str, Any] | None = None,  # noqa: N803
    ) -> None:
        self.uploads.append((Filename, Bucket, Key, ExtraArgs))
        self.snapshots[Key] = Path(Filename).read_bytes()


@pytest.fixture
def s3() -> Any:
    fake = FakeS3()
    publish.use_client(fake)
    yield fake
    publish.use_client(None)


def test_split_and_is_s3() -> None:
    assert publish.is_s3("s3://b/k") and not publish.is_s3("https://b/k")
    assert publish.split("s3://bucket/some/key.mp4") == ("bucket", "some/key.mp4")
    assert publish.split("s3://bucket") == ("bucket", "")
    with pytest.raises(ValueError):
        publish.split("https://example.org/x")


def test_download_writes_file_and_checksum(s3: FakeS3, tmp_path: Path) -> None:
    s3.objects["src/videos/a.mp4"] = b"video bytes"
    dst = publish.download("s3://src/videos/a.mp4", tmp_path / "work" / "a.mp4")
    assert dst.read_bytes() == b"video bytes"
    assert (tmp_path / "work" / "a.mp4.sha256").read_text().strip()
    assert not (tmp_path / "work" / "a.mp4.part").exists()


def test_ingest_download_routes_s3_urls_to_the_client(s3: FakeS3, tmp_path: Path) -> None:
    s3.objects["src/a.mp4"] = b"x"
    ingest._download("s3://src/a.mp4", tmp_path / "a.mp4")
    assert (tmp_path / "a.mp4").read_bytes() == b"x"


def test_absolutise_rewrites_the_four_url_fields_and_leaves_absolute_ones() -> None:
    manifest: dict[str, Any] = {
        "video": {"main": {"url": "main.mp4"}, "poster": "poster.jpg"},
        "signLanguage": [{"track": {"url": "signer-bfi.mp4"}}],
        "captions": [
            {"url": "captions/verbatim.en.vtt"},
            {"url": "https://elsewhere.example/simplified.vtt"},
        ],
    }
    out = publish.absolutise(manifest, "https://cdn.example/titles/t1/")
    assert out["video"]["main"]["url"] == "https://cdn.example/titles/t1/main.mp4"
    assert out["video"]["poster"] == "https://cdn.example/titles/t1/poster.jpg"
    assert out["signLanguage"][0]["track"]["url"] == "https://cdn.example/titles/t1/signer-bfi.mp4"
    assert out["captions"][0]["url"] == "https://cdn.example/titles/t1/captions/verbatim.en.vtt"
    assert out["captions"][1]["url"] == "https://elsewhere.example/simplified.vtt"
    assert manifest["video"]["main"]["url"] == "main.mp4"  # the input is untouched


def test_upload_title_maps_paths_types_and_rewrites_the_manifest(
    s3: FakeS3, tmp_path: Path
) -> None:
    title = tmp_path / "out" / "t1"
    (title / "captions").mkdir(parents=True)
    (title / "main.mp4").write_bytes(b"m")
    (title / "poster.jpg").write_bytes(b"p")
    (title / "captions" / "verbatim.en.vtt").write_text("WEBVTT\n")
    (title / "analysis.json").write_text("{}")
    manifest = {"video": {"main": {"url": "main.mp4"}, "poster": "poster.jpg"}, "captions": []}
    (title / "manifest.json").write_text(json.dumps(manifest))

    urls = publish.upload_title(title, "s3://assets/titles", "https://cdn.example/titles")

    keys = [k for _, _, k, _ in s3.uploads]
    assert keys == [
        "titles/t1/captions/verbatim.en.vtt",
        "titles/t1/main.mp4",
        "titles/t1/manifest.json",
        "titles/t1/poster.jpg",
    ]
    assert "titles/t1/analysis.json" not in keys
    assert urls[1] == "s3://assets/titles/t1/main.mp4"
    types = {k: (x or {}).get("ContentType") for _, _, k, x in s3.uploads}
    assert types["titles/t1/main.mp4"] == "video/mp4"
    assert types["titles/t1/captions/verbatim.en.vtt"] == "text/vtt"
    assert types["titles/t1/manifest.json"] == "application/json"
    uploaded = json.loads(s3.snapshots["titles/t1/manifest.json"])
    assert uploaded["video"]["main"]["url"] == "https://cdn.example/titles/t1/main.mp4"
    # the local manifest keeps its relative URLs, and the temp copy is gone
    assert json.loads((title / "manifest.json").read_text())["video"]["main"]["url"] == "main.mp4"
    assert not (title / "manifest.public.json").exists()


def test_upload_title_without_prefix_or_public_base(s3: FakeS3, tmp_path: Path) -> None:
    title = tmp_path / "t2"
    title.mkdir()
    (title / "manifest.json").write_text('{"video": {"main": {"url": "main.mp4"}}}')
    publish.upload_title(title, "s3://assets")
    _, bucket, key, _ = s3.uploads[0]
    assert (bucket, key) == ("assets", "t2/manifest.json")
    assert json.loads(s3.snapshots[key])["video"]["main"]["url"] == "main.mp4"


def test_fetch_media_prefers_the_media_field_over_the_publisher_url(
    s3: FakeS3, tmp_path: Path
) -> None:
    s3.objects["sources/sp-b-full.mp4"] = b"video"
    entry = ingest.parse_entry(
        {
            "id": "t3",
            "title": "T",
            "url": "https://www.scottishparliament.tv/meeting/x",
            "media": "s3://sources/sp-b-full.mp4",
            "publisher": "P",
            "licence": "L",
            "attribution": "A",
        }
    )
    assert entry.url.startswith("https://")  # the manifest's source stays the publisher
    path = ingest.fetch_media(entry, tmp_path / "work")
    assert path.read_bytes() == b"video"
    assert path.name == "source.mp4"
