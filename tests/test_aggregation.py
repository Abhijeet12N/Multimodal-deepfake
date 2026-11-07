"""Unit tests for score aggregation and suspicious interval merging."""
from __future__ import annotations

import numpy as np
import pytest

from deepfake_detector.audio_detection.aggregation import (
    RecordingScoreAggregator,
    merge_suspicious_intervals,
)


class TestRecordingScoreAggregator:
    def test_empty_scores(self):
        agg = RecordingScoreAggregator(strategy="mean")
        assert agg.aggregate([]) == 0.0

    def test_mean_strategy(self):
        agg = RecordingScoreAggregator(strategy="mean")
        scores = [0.2, 0.4, 0.6, 0.8]
        assert np.isclose(agg.aggregate(scores), 0.5)

    def test_top_k_strategy(self):
        agg = RecordingScoreAggregator(strategy="top_k", top_k=2)
        scores = [0.1, 0.9, 0.3, 0.7, 0.2]
        # Top 2 are 0.9 and 0.7 -> mean = 0.8
        assert np.isclose(agg.aggregate(scores), 0.8)

    def test_top_k_exceeding_length(self):
        agg = RecordingScoreAggregator(strategy="top_k", top_k=10)
        scores = [0.2, 0.6]
        # k=10 > len(scores)=2 -> uses all 2 -> mean = 0.4
        assert np.isclose(agg.aggregate(scores), 0.4)

    def test_percentile_strategy(self):
        agg_90 = RecordingScoreAggregator(strategy="percentile", percentile=90.0)
        scores = list(np.linspace(0.0, 1.0, 101))
        assert np.isclose(agg_90.aggregate(scores), 0.90, atol=1e-2)

        agg_50 = RecordingScoreAggregator(strategy="percentile", percentile=50.0)
        assert np.isclose(agg_50.aggregate(scores), 0.50, atol=1e-2)

    def test_duration_weighted_strategy(self):
        agg = RecordingScoreAggregator(strategy="duration_weighted")
        scores = [0.2, 0.8]
        durations = [1.0, 3.0]
        # Weighted mean: (0.2*1 + 0.8*3) / 4.0 = 2.6 / 4.0 = 0.65
        assert np.isclose(agg.aggregate(scores, durations), 0.65)

    def test_duration_weighted_fallback(self):
        agg = RecordingScoreAggregator(strategy="duration_weighted")
        scores = [0.2, 0.8]
        # Fallback to mean when durations are None or mismatched
        assert np.isclose(agg.aggregate(scores, None), 0.5)
        assert np.isclose(agg.aggregate(scores, [1.0]), 0.5)


class TestMergeSuspiciousIntervals:
    def test_empty_windows(self):
        assert merge_suspicious_intervals([]) == []

    def test_single_window(self):
        win = [{"start_seconds": 1.0, "end_seconds": 3.0, "fake_score": 0.85}]
        intervals = merge_suspicious_intervals(win)
        assert len(intervals) == 1
        assert intervals[0]["start_seconds"] == 1.0
        assert intervals[0]["end_seconds"] == 3.0
        assert intervals[0]["duration_seconds"] == 2.0
        assert intervals[0]["peak_fake_score"] == 0.85
        assert intervals[0]["mean_fake_score"] == 0.85
        assert intervals[0]["supporting_windows"] == 1

    def test_overlapping_windows_merged(self):
        wins = [
            {"start_seconds": 0.0, "end_seconds": 2.0, "fake_score": 0.70},
            {"start_seconds": 1.0, "end_seconds": 3.0, "fake_score": 0.90},
            {"start_seconds": 2.0, "end_seconds": 4.0, "fake_score": 0.80},
        ]
        intervals = merge_suspicious_intervals(wins)
        assert len(intervals) == 1
        assert intervals[0]["start_seconds"] == 0.0
        assert intervals[0]["end_seconds"] == 4.0
        assert intervals[0]["duration_seconds"] == 4.0
        assert intervals[0]["peak_fake_score"] == 0.90
        assert np.isclose(intervals[0]["mean_fake_score"], 0.80)
        assert intervals[0]["supporting_windows"] == 3

    def test_disjoint_windows_separate(self):
        wins = [
            {"start_seconds": 0.0, "end_seconds": 2.0, "fake_score": 0.80},
            # Gap of 3 seconds (> default tolerance 0.5s)
            {"start_seconds": 5.0, "end_seconds": 7.0, "fake_score": 0.95},
        ]
        intervals = merge_suspicious_intervals(wins, tolerance_seconds=0.5)
        assert len(intervals) == 2
        assert intervals[0]["start_seconds"] == 0.0
        assert intervals[0]["end_seconds"] == 2.0
        assert intervals[1]["start_seconds"] == 5.0
        assert intervals[1]["end_seconds"] == 7.0

    def test_tolerance_gap_bridging(self):
        wins = [
            {"start_seconds": 1.0, "end_seconds": 2.5, "fake_score": 0.75},
            # Gap is 0.3s <= tolerance 0.5s
            {"start_seconds": 2.8, "end_seconds": 4.0, "fake_score": 0.85},
        ]
        intervals = merge_suspicious_intervals(wins, tolerance_seconds=0.5)
        assert len(intervals) == 1
        assert intervals[0]["start_seconds"] == 1.0
        assert intervals[0]["end_seconds"] == 4.0
        assert intervals[0]["supporting_windows"] == 2

    def test_out_of_order_windows_sorted(self):
        wins = [
            {"start_seconds": 4.0, "end_seconds": 6.0, "fake_score": 0.70},
            {"start_seconds": 0.0, "end_seconds": 2.0, "fake_score": 0.90},
        ]
        intervals = merge_suspicious_intervals(wins)
        assert len(intervals) == 2
        assert intervals[0]["start_seconds"] == 0.0
        assert intervals[1]["start_seconds"] == 4.0
