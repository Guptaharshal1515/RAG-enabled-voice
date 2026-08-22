from typing import List, Dict, Any, Union
import numpy as np


def compute_percentiles(
    values: List[Union[int, float]],
    percentiles: List[int] = [50, 70, 90, 95, 99, 100]
) -> Dict[str, float]:
    """
    Computes statistical percentiles, mean, min, and max for a list of latency values.
    """
    if not values:
        return {f"P{p}": 0.0 for p in percentiles} | {"mean": 0.0, "min": 0.0, "max": 0.0, "count": 0}

    arr = np.array(values, dtype=np.float64)
    stats: Dict[str, float] = {}

    for p in percentiles:
        stats[f"P{p}"] = round(float(np.percentile(arr, p)), 2)

    stats["mean"] = round(float(np.mean(arr)), 2)
    stats["min"] = round(float(np.min(arr)), 2)
    stats["max"] = round(float(np.max(arr)), 2)
    stats["count"] = len(values)

    return stats


def compute_stage_breakdown(
    records: List[Dict[str, Any]]
) -> Dict[str, Dict[str, float]]:
    """
    Computes percentile distributions across all distinct numeric pipeline stages.
    """
    if not records:
        return {}

    stages = set()
    for r in records:
        for k, v in r.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                stages.add(k)

    stage_summary = {}
    for stage in sorted(stages):
        stage_vals = [
            float(r[stage])
            for r in records
            if stage in r and isinstance(r[stage], (int, float)) and not isinstance(r[stage], bool)
        ]
        if stage_vals:
            stage_summary[stage] = compute_percentiles(stage_vals)

    return stage_summary
