from __future__ import annotations

from typing import Protocol

from stm_pipeline.types import Box, SampledFrame


class Detector(Protocol):
    """Finds people in a frame. Boxes are in the sampled frame's pixel coordinates."""

    name: str

    def detect(self, frame: SampledFrame) -> list[Box]: ...
