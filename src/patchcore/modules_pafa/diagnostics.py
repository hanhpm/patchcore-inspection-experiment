"""Diagnostics for PAFA experiment checkpoints."""

from typing import Any
from typing import Dict
from typing import Sequence

import numpy as np


def compute_e3_diagnostics(
    scores: Sequence[float],
    anomaly_labels: Sequence[bool],
    adapter_displacement: float,
    aupro_005: float,
) -> Dict[str, Any]:
    """Compute the four required PAFA E3 diagnostics."""
    scores_array = np.asarray(scores, dtype=float)
    labels_array = np.asarray(anomaly_labels, dtype=bool)
    nominal_scores = scores_array[~labels_array]
    anomaly_scores = scores_array[labels_array]

    nominal_p95 = _percentile_or_none(nominal_scores, 95)
    nominal_p99 = _percentile_or_none(nominal_scores, 99)
    real_gt_margin = None
    if len(nominal_scores) > 0 and len(anomaly_scores) > 0:
        real_gt_margin = float(np.mean(anomaly_scores) - np.mean(nominal_scores))

    return {
        "adapter_displacement": float(adapter_displacement),
        "nominal_stability": {
            "score_p95": nominal_p95,
            "score_p99": nominal_p99,
            "source": "test_nominal_images",
        },
        "real_gt_margin": real_gt_margin,
        "real_gt_margin_definition": (
            "mean(anomaly_image_scores) - mean(nominal_image_scores)"
        ),
        "au_pro_0.05": float(aupro_005),
    }


def _percentile_or_none(values: np.ndarray, percentile: float):
    if len(values) == 0:
        return None
    return float(np.percentile(values, percentile))
