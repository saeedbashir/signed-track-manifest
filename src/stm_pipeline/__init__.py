"""Offline extraction pipeline that turns a flattened signed video into STM layers.

It finds a person and cuts a rectangle. It does not recognise signs, interpret
meaning, assess interpretation quality, or generate anything.
"""

from stm_pipeline.captions_vtt import Cue, build_vtt, parse_vtt
from stm_pipeline.config import IdentifyWeights, PipelineConfig
from stm_pipeline.simplify import BedrockSimplifier, NullSimplifier, Simplifier
from stm_pipeline.types import Box, FrameDetections, FrameInfo, SampledFrame

__all__ = [
    "BedrockSimplifier",
    "Box",
    "Cue",
    "FrameDetections",
    "FrameInfo",
    "IdentifyWeights",
    "NullSimplifier",
    "PipelineConfig",
    "SampledFrame",
    "Simplifier",
    "build_vtt",
    "parse_vtt",
]
