"""Validation harness: ground-truth rectangles, intersection-over-union scoring."""

from stm_pipeline.harness.ground_truth import GroundTruth, load_ground_truth, save_ground_truth
from stm_pipeline.harness.score import ClipScore, Report, score

__all__ = ["ClipScore", "GroundTruth", "Report", "load_ground_truth", "save_ground_truth", "score"]
