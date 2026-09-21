"""Extraction confidence: one public number from 0 to 1.

Combines detection rate (persistence), positional stability and motion energy.
Below :data:`stm.model.LOW_CONFIDENCE_THRESHOLD` the title is flagged in the
manifest and badged in any UI. A wrong crop that ships silently is worse than
one the viewer was warned about.
"""

from __future__ import annotations

from stm_pipeline.identify import ClusterFeatures

_W_PERSISTENCE = 0.4
_W_STABILITY = 0.3
_W_MOTION = 0.3


def confidence(f: ClusterFeatures) -> float:
    value = _W_PERSISTENCE * f.persistence + _W_STABILITY * f.stability + _W_MOTION * f.motion_score
    return max(0.0, min(1.0, value))
