"""Unit tests for the YOLOX pre/post-processing that need no weights, plus one
integration test that does."""

from __future__ import annotations

import numpy as np
import pytest

from stm_pipeline.detect.onnx_person import decode, letterbox, nms
from tests.conftest import integration


def test_letterbox_pads_top_left_with_grey() -> None:
    img = np.zeros((90, 160, 3), dtype=np.uint8)
    chw, ratio = letterbox(img, 416)
    assert chw.shape == (3, 416, 416)
    assert chw.dtype == np.float32
    assert ratio == pytest.approx(416 / 160)
    # padded rows below the resized image are grey
    assert chw[:, 300, 0].tolist() == [114.0, 114.0, 114.0]
    # image content is black at the top-left
    assert chw[:, 0, 0].tolist() == [0.0, 0.0, 0.0]


def test_decode_grid_count_and_scaling() -> None:
    size = 416
    n = sum((size // s) ** 2 for s in (8, 16, 32))
    raw = np.zeros((n, 85), dtype=np.float32)
    out = decode(raw, size)
    assert out.shape == (n, 85)
    # first cell of the stride-8 grid decodes to centre (0,0) and size 8x8 (exp(0)*8)
    assert out[0, :4].tolist() == [0.0, 0.0, 8.0, 8.0]
    # last cell of the stride-32 grid sits at the far corner
    assert out[-1, 0] == pytest.approx((size // 32 - 1) * 32)


def test_nms_suppresses_overlaps() -> None:
    boxes = np.array([[0, 0, 100, 100], [5, 5, 105, 105], [200, 200, 300, 300]], dtype=np.float32)
    scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
    assert nms(boxes, scores, 0.45) == [0, 2]


@integration
def test_real_model_runs_on_a_frame() -> None:
    from stm_pipeline.detect.onnx_person import OnnxPersonDetector
    from stm_pipeline.types import SampledFrame

    det = OnnxPersonDetector("yolox-tiny")
    frame = SampledFrame(0, 0.0, np.zeros((360, 640, 3), dtype=np.uint8), 1.0)
    assert det.detect(frame) == []
