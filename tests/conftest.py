from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

from stm_pipeline import ffmpeg

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def load_example(name: str) -> dict[str, Any]:
    data: dict[str, Any] = json.loads((EXAMPLES / name).read_text(encoding="utf-8"))
    return data


requires_ffmpeg = pytest.mark.skipif(not ffmpeg.available(), reason="ffmpeg not on PATH")
integration = pytest.mark.skipif(
    os.environ.get("STM_RUN_INTEGRATION") != "1",
    reason="set STM_RUN_INTEGRATION=1 to run tests that need model weights",
)


@pytest.fixture(scope="session")
def synthetic_clip(tmp_path_factory: pytest.TempPathFactory) -> Path:
    if not ffmpeg.available():
        pytest.skip("ffmpeg not on PATH")
    path = tmp_path_factory.mktemp("media") / "synthetic.mp4"
    ffmpeg.synthetic_clip(path, width=320, height=180, seconds=2.0, fps=25)
    return path
