"""A scripted detector for tests: returns whatever boxes the script says for a frame index."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

from stm_pipeline.types import Box, SampledFrame


class ScriptedDetector:
    name = "scripted"

    def __init__(
        self,
        script: Mapping[int, Sequence[Box]] | Callable[[int], Sequence[Box]],
    ) -> None:
        self._script = script

    def detect(self, frame: SampledFrame) -> list[Box]:
        if callable(self._script):
            return list(self._script(frame.index))
        return list(self._script.get(frame.index, ()))
