"""Aggregation and interval merging module.

Aggregates window-level scores into recording-level decisions and detects
suspicious contiguous intervals with exact start and end timestamps.
"""
from __future__ import annotations

from typing import Any, List, Optional
import numpy as np


def merge_suspicious_intervals(
    flagged_windows: list[dict[str, Any]],
    tolerance_seconds: float = 0.5,
) -> list[dict[str, Any]]:
    """Merge overlapping or contiguous flagged windows into continuous suspicious intervals.

    Parameters
    ----------
    flagged_windows : list[dict[str, Any]]
        List of window dictionaries containing 'start_seconds', 'end_seconds', 'fake_score'.
    tolerance_seconds : float
        Maximum gap between windows to treat them as contiguous.

    Returns
    -------
    list[dict[str, Any]]
        List of merged interval dicts with 'start_seconds', 'end_seconds', 'peak_score', 'mean_score'.
    """
    if not flagged_windows:
        return []

    # Sort windows chronologically
    sorted_windows = sorted(flagged_windows, key=lambda w: w["start_seconds"])
    intervals: list[dict[str, Any]] = []

    current_start = sorted_windows[0]["start_seconds"]
    current_end = sorted_windows[0]["end_seconds"]
    current_scores = [sorted_windows[0].get("fake_score", 0.0)]

    for win in sorted_windows[1:]:
        if win["start_seconds"] <= current_end + tolerance_seconds:
            # Overlap or continuous: extend interval
            current_end = max(current_end, win["end_seconds"])
            current_scores.append(win.get("fake_score", 0.0))
        else:
            # Gap: finalize current interval and begin a new one
            intervals.append({
                "start_seconds": round(current_start, 2),
                "end_seconds": round(current_end, 2),
                "duration_seconds": round(current_end - current_start, 2),
                "peak_fake_score": round(float(np.max(current_scores)), 4),
                "mean_fake_score": round(float(np.mean(current_scores)), 4),
                "supporting_windows": len(current_scores),
            })
            current_start = win["start_seconds"]
            current_end = win["end_seconds"]
            current_scores = [win.get("fake_score", 0.0)]

    intervals.append({
        "start_seconds": round(current_start, 2),
        "end_seconds": round(current_end, 2),
        "duration_seconds": round(current_end - current_start, 2),
        "peak_fake_score": round(float(np.max(current_scores)), 4),
        "mean_fake_score": round(float(np.mean(current_scores)), 4),
        "supporting_windows": len(current_scores),
    })

    return intervals


class RecordingScoreAggregator:
    """Aggregates window-level prediction scores into a recording-level score."""

    def __init__(
        self,
        strategy: str = "mean",  # "mean", "top_k", "percentile", "duration_weighted"
        top_k: int = 3,
        percentile: float = 90.0,
    ):
        self.strategy = strategy
        self.top_k = top_k
        self.percentile = percentile

    def aggregate(
        self,
        scores: list[float],
        durations: list[float] | None = None,
    ) -> float:
        """Aggregate list of window scores into a single recording score."""
        if not scores:
            return 0.0

        arr = np.asarray(scores, dtype=np.float64)

        if self.strategy == "top_k":
            k = min(self.top_k, len(arr))
            top_vals = np.sort(arr)[-k:]
            return float(np.mean(top_vals))

        elif self.strategy == "percentile":
            return float(np.percentile(arr, self.percentile))

        elif self.strategy == "duration_weighted" and durations is not None and len(durations) == len(scores):
            d_arr = np.asarray(durations, dtype=np.float64)
            weights = d_arr / np.sum(d_arr).clip(min=1e-6)
            return float(np.sum(arr * weights))

        # Default: mean
        return float(np.mean(arr))
