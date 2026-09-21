"""Frame sampling: every Nth frame, downscaled, streamed one at a time."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt

from stm_pipeline.types import SampledFrame


def iter_sampled_frames(path: Path, every: int = 5, max_width: int = 640) -> Iterator[SampledFrame]:
    """Yield every ``every``-th frame, resized so width <= ``max_width``.

    Frames are grabbed without decoding when skipped, which is what makes a
    20-minute source affordable. Nothing is kept in memory between frames.
    """
    if every < 1:
        raise ValueError("every must be >= 1")
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise OSError(f"could not open video: {path}")
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 0.0
        index = 0
        while True:
            if not cap.grab():
                break
            if index % every == 0:
                ok, frame = cap.retrieve()
                if not ok or frame is None:
                    break
                image: npt.NDArray[np.uint8] = np.asarray(frame, dtype=np.uint8)
                scale = 1.0
                if image.shape[1] > max_width:
                    scale = max_width / image.shape[1]
                    new_h = max(1, round(image.shape[0] * scale))
                    image = np.asarray(
                        cv2.resize(image, (max_width, new_h), interpolation=cv2.INTER_AREA),
                        dtype=np.uint8,
                    )
                time_s = index / fps if fps > 0 else 0.0
                yield SampledFrame(index=index, time_s=time_s, image=image, scale=scale)
            index += 1
    finally:
        cap.release()
