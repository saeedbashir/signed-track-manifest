"""YOLOX person detector on onnxruntime.

Preprocessing and decoding follow the YOLOX ONNXRuntime demo exactly: letterbox
to a square input anchored top-left with grey (114) padding, BGR, no
normalisation; outputs decoded with strides 8/16/32; score = objectness × class
probability; class-agnostic NMS. Only COCO class 0 (person) is returned.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt

from stm_pipeline.detect.models import MODELS, ensure_model
from stm_pipeline.types import Box, SampledFrame

_PERSON_CLASS = 0
_STRIDES = (8, 16, 32)


def letterbox(image: npt.NDArray[np.uint8], size: int) -> tuple[npt.NDArray[np.float32], float]:
    """Resize keeping aspect, pad bottom/right with 114. Returns CHW float32 and the ratio."""
    padded = np.full((size, size, 3), 114, dtype=np.uint8)
    r = min(size / image.shape[0], size / image.shape[1])
    new_w = int(image.shape[1] * r)
    new_h = int(image.shape[0] * r)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    padded[:new_h, :new_w] = resized
    chw = np.ascontiguousarray(padded.transpose(2, 0, 1), dtype=np.float32)
    return chw, r


def decode(raw: npt.NDArray[np.float32], size: int) -> npt.NDArray[np.float32]:
    """Turn raw YOLOX head output (N, 5 + classes) into (cx, cy, w, h, obj, cls...) in pixels."""
    grids = []
    strides = []
    for stride in _STRIDES:
        hsize = wsize = size // stride
        xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
        grid = np.stack((xv, yv), 2).reshape(-1, 2)
        grids.append(grid)
        strides.append(np.full((grid.shape[0], 1), stride))
    grid_all = np.concatenate(grids, 0).astype(np.float32)
    stride_all = np.concatenate(strides, 0).astype(np.float32)
    out = raw.copy()
    out[:, :2] = (out[:, :2] + grid_all) * stride_all
    out[:, 2:4] = np.exp(out[:, 2:4]) * stride_all
    return out


def nms(boxes: npt.NDArray[np.float32], scores: npt.NDArray[np.float32], thr: float) -> list[int]:
    """Greedy NMS on (x1, y1, x2, y2) boxes. Returns kept indices."""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[np.where(ovr <= thr)[0] + 1]
    return keep


class OnnxPersonDetector:
    """YOLOX on onnxruntime, returning person boxes in the input frame's coordinates."""

    def __init__(
        self,
        model: str = "yolox-tiny",
        model_path: Path | None = None,
        score_threshold: float = 0.35,
        nms_threshold: float = 0.45,
        cache_dir: Path | None = None,
    ) -> None:
        import onnxruntime as ort

        spec = MODELS[model]
        self.name = spec.name
        self.input_size = spec.input_size
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        path = model_path or ensure_model(model, cache_dir)
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        self._session = ort.InferenceSession(
            str(path), sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self._input_name = self._session.get_inputs()[0].name

    def detect(self, frame: SampledFrame) -> list[Box]:
        chw, ratio = letterbox(frame.image, self.input_size)
        raw = self._session.run(None, {self._input_name: chw[None, :, :, :]})[0][0]
        pred = decode(np.asarray(raw, dtype=np.float32), self.input_size)
        obj = pred[:, 4]
        cls_scores = pred[:, 5:]
        person = obj * cls_scores[:, _PERSON_CLASS]
        mask = person > self.score_threshold
        if not mask.any():
            return []
        p = pred[mask]
        s = person[mask]
        xyxy = np.empty((p.shape[0], 4), dtype=np.float32)
        xyxy[:, 0] = p[:, 0] - p[:, 2] / 2.0
        xyxy[:, 1] = p[:, 1] - p[:, 3] / 2.0
        xyxy[:, 2] = p[:, 0] + p[:, 2] / 2.0
        xyxy[:, 3] = p[:, 1] + p[:, 3] / 2.0
        xyxy /= ratio
        keep = nms(xyxy, s, self.nms_threshold)
        w, h = frame.width, frame.height
        boxes: list[Box] = []
        for i in keep:
            x1 = float(np.clip(xyxy[i, 0], 0, w))
            y1 = float(np.clip(xyxy[i, 1], 0, h))
            x2 = float(np.clip(xyxy[i, 2], 0, w))
            y2 = float(np.clip(xyxy[i, 3], 0, h))
            if x2 - x1 < 2 or y2 - y1 < 2:
                continue
            boxes.append(Box(x1, y1, x2, y2, float(s[i])))
        return boxes
