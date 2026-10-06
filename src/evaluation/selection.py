"""Choose between the stage-1 and stage-2 models using VALIDATION metrics only."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


def select_stage(validation_metrics: dict[int, dict[str, Any]], rule: dict) -> dict[str, Any]:
    """Return the selection decision.

    Rule (from config ``training.model_selection``): maximise ``metric``
    (default macro recall = mean per-class sensitivity, robust to class
    imbalance and aligned with a screening objective of not missing any
    class); if within ``tie_tolerance``, prefer the lower ``tie_breaker``
    (validation loss).
    """
    metric, tie_breaker = rule["metric"], rule.get("tie_breaker", "loss")
    tolerance = float(rule.get("tie_tolerance", 1e-6))
    stages = sorted(validation_metrics)
    best = stages[0]
    for stage in stages[1:]:
        delta = validation_metrics[stage][metric] - validation_metrics[best][metric]
        if delta > tolerance or (
            abs(delta) <= tolerance and validation_metrics[stage][tie_breaker] < validation_metrics[best][tie_breaker]
        ):
            best = stage
    return {
        "selected_stage": best,
        "criterion": f"max validation {metric}; tie → min validation {tie_breaker}",
        "validation": {
            str(s): {k: validation_metrics[s][k] for k in (metric, tie_breaker, "accuracy", "macro_f1")}
            for s in stages
        },
        "decided_at_utc": datetime.now(timezone.utc).isoformat(),
        "test_set_used": False,
    }
