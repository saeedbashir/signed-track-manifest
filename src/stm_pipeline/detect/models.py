"""Model registry: where weights come from, their checksum, and their licence.

Only permissively licensed models belong here. This repository ships under
Apache 2.0, so anything AGPL — the ``ultralytics`` YOLOv8 family, for instance —
is out regardless of how good it is. Check the weights licence separately from
the code licence; they are often different.
"""

from __future__ import annotations

import hashlib
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelSpec:
    name: str
    url: str
    sha256: str
    input_size: int
    licence: str
    source: str


_YOLOX_RELEASE = "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0"

MODELS: dict[str, ModelSpec] = {
    "yolox-tiny": ModelSpec(
        name="yolox-tiny",
        url=f"{_YOLOX_RELEASE}/yolox_tiny.onnx",
        sha256="427cc366d34e27ff7a03e2899b5e3671425c262ea2291f88bb942bc1cc70b0f7",
        input_size=416,
        licence="Apache-2.0 (release asset of the YOLOX repository; no separate weights licence)",
        source="https://github.com/Megvii-BaseDetection/YOLOX",
    ),
    "yolox-nano": ModelSpec(
        name="yolox-nano",
        url=f"{_YOLOX_RELEASE}/yolox_nano.onnx",
        sha256="c789161ed43c8269fcd4e67c67eeeb4e80c622da2eb296a20bc6007bd18a0b7d",
        input_size=416,
        licence="Apache-2.0 (release asset of the YOLOX repository; no separate weights licence)",
        source="https://github.com/Megvii-BaseDetection/YOLOX",
    ),
}


def model_dir() -> Path:
    env = os.environ.get("STM_MODEL_DIR")
    if env:
        return Path(env)
    return Path.home() / ".cache" / "stm" / "models"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_model(name: str, cache_dir: Path | None = None) -> Path:
    """Return a local path to the model, downloading and checksum-verifying on first use."""
    if name not in MODELS:
        raise KeyError(f"unknown model {name!r}; known: {', '.join(sorted(MODELS))}")
    spec = MODELS[name]
    target_dir = cache_dir or model_dir()
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / Path(spec.url).name
    if target.exists() and _sha256(target) == spec.sha256:
        return target
    tmp = target.with_suffix(".part")
    urllib.request.urlretrieve(spec.url, tmp)
    digest = _sha256(tmp)
    if digest != spec.sha256:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(
            f"checksum mismatch for {spec.name}: expected {spec.sha256}, got {digest}"
        )
    tmp.replace(target)
    return target
