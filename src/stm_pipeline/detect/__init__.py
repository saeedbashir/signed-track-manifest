"""Person detectors behind one protocol: a sampled frame in, boxes out."""

from stm_pipeline.detect.base import Detector
from stm_pipeline.detect.mock import ScriptedDetector

__all__ = ["Detector", "ScriptedDetector"]
