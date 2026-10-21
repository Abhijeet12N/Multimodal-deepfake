"""Unit tests for Phase 2 Evaluation Pipeline.

Covers:
- scripts/build_manifest.py: schema validation, label conventions, dataset summary
- scripts/check_data_leakage.py: speaker overlap, source ID overlap, duplicate detection
- scripts/evaluate_audio.py: EER calculation, score orientations, metrics, per-attack breakdowns
"""
from __future__ import annotations

import csv
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from build_manifest import (
    MANIFEST_COLUMNS,
    generate_dataset_summary,
    validate_manifest_row,
    write_manifest_csv,
)
from check_data_leakage import (
    check_sample_and_hash_duplicates,
    check_source_id_leakage,
    check_speaker_leakage,
    run_full_leakage_audit,
)
from evaluate_audio import (
    compute_eer,
    compute_per_attack_metrics,
    evaluate_predictions,
)


class TestBuildManifest(unittest.TestCase):
    """Test suite for manifest construction and schema validation."""

    def test_valid_manifest_row(self):
        row = {
            "sample_id": "test_01",
            "path": "/path/to/test_01.wav",
            "label": 0,
            "dataset": "test_ds",
            "split": "dev",
            "speaker_id": "spk_1",
            "source_id": "test_01",
            "attack_type": "-",
            "language": "en",
        }
        # Should not raise
        validate_manifest_row(row)

    def test_invalid_label_raises_error(self):
        row = {
            "sample_id": "test_01",
            "path": "/path/to/test_01.wav",
            "label": 2,  # Invalid: only 0 or 1 allowed
            "dataset": "test_ds",
            "split": "dev",
            "speaker_id": "spk_1",
            "source_id": "test_01",
            "attack_type": "-",
            "language": "en",
        }
        with self.assertRaises(ValueError):
            validate_manifest_row(row)

    def test_missing_column_raises_error(self):
        row = {
            "sample_id": "test_01",
            "path": "/path/to/test_01.wav",
            "label": 1,
            # missing split, speaker_id, etc.
        }
        with self.assertRaises(ValueError):
            validate_manifest_row(row)

    def test_write_and_read_manifest(self):
        rows = [
            {
                "sample_id": "utt_01",
                "path": "audio1.wav",
                "label": 0,
                "dataset": "ds_a",
                "split": "train",
                "speaker_id": "spk_01",
                "source_id": "utt_01",
                "attack_type": "-",
                "language": "en",
            },
            {
                "sample_id": "utt_02",
                "path": "audio2.wav",
                "label": 1,
                "dataset": "ds_a",
                "split": "train",
                "speaker_id": "spk_02",
                "source_id": "utt_02",
                "attack_type": "A01",
                "language": "en",
            },
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            out_csv = Path(tmpdir) / "test_manifest.csv"
            write_manifest_csv(rows, out_csv)
            self.assertTrue(out_csv.exists())

            with open(out_csv, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                read_rows = list(reader)
                self.assertEqual(len(read_rows), 2)
                self.assertEqual(list(read_rows[0].keys()), MANIFEST_COLUMNS)
                self.assertEqual(int(read_rows[0]["label"]), 0)
                self.assertEqual(int(read_rows[1]["label"]), 1)

    def test_generate_dataset_summary(self):
        rows = [
            {"sample_id": "1", "path": "", "label": 0, "dataset": "d1", "split": "dev", "speaker_id": "s1", "source_id": "1", "attack_type": "-", "language": "en"},
            {"sample_id": "2", "path": "", "label": 1, "dataset": "d1", "split": "dev", "speaker_id": "s2", "source_id": "2", "attack_type": "A01", "language": "en"},
            {"sample_id": "3", "path": "", "label": 1, "dataset": "d1", "split": "eval", "speaker_id": "s3", "source_id": "3", "attack_type": "A07", "language": "en"},
        ]
        summary = generate_dataset_summary(rows)
        self.assertEqual(summary["total_samples"], 3)
        self.assertEqual(summary["labels"]["genuine_count"], 1)
        self.assertEqual(summary["labels"]["fake_count"], 2)
        self.assertAlmostEqual(summary["labels"]["fake_ratio"], 0.6667, places=3)
        self.assertEqual(summary["splits"]["dev"]["total"], 2)
        self.assertEqual(summary["splits"]["eval"]["total"], 1)
        self.assertIn("A01", summary["attack_types"])
        self.assertIn("A07", summary["attack_types"])


class TestLeakageAudit(unittest.TestCase):
    """Test suite for data leakage checks."""

    def test_disjoint_speakers_pass(self):
        data = {
            "train": [{"speaker_id": "spk_1"}, {"speaker_id": "spk_2"}],
            "dev": [{"speaker_id": "spk_3"}, {"speaker_id": "spk_4"}],
            "eval": [{"speaker_id": "spk_5"}],
        }
        res = check_speaker_leakage(data)
        self.assertEqual(res["status"], "PASS")
        self.assertFalse(res["has_speaker_leakage"])

    def test_overlapping_speakers_fail(self):
        data = {
            "train": [{"speaker_id": "spk_1"}, {"speaker_id": "spk_2"}],
            "dev": [{"speaker_id": "spk_2"}, {"speaker_id": "spk_3"}],  # spk_2 leaked!
        }
        res = check_speaker_leakage(data)
        self.assertEqual(res["status"], "FAIL")
        self.assertTrue(res["has_speaker_leakage"])
        self.assertEqual(len(res["overlaps"]), 1)
        self.assertIn("spk_2", res["overlaps"][0]["overlapping_speakers"])

    def test_source_id_leakage(self):
        data_clean = {
            "train": [{"source_id": "rec_01"}, {"source_id": "rec_02"}],
            "dev": [{"source_id": "rec_03"}],
        }
        self.assertEqual(check_source_id_leakage(data_clean)["status"], "PASS")

        data_leaked = {
            "train": [{"source_id": "rec_01"}],
            "dev": [{"source_id": "rec_01"}],  # derivative in dev!
        }
        res = check_source_id_leakage(data_leaked)
        self.assertEqual(res["status"], "FAIL")
        self.assertTrue(res["has_source_leakage"])

    def test_sample_id_duplicates(self):
        data = {
            "train": [{"sample_id": "utt_100"}],
            "dev": [{"sample_id": "utt_100"}],
        }
        res = check_sample_and_hash_duplicates(data)
        self.assertEqual(res["status"], "FAIL")
        self.assertTrue(res["has_duplicate_samples"])


class TestEvaluationMetrics(unittest.TestCase):
    """Test suite for metrics calculation."""

    def test_compute_eer_perfect_separation(self):
        # Genuine scores low (0.0 to 0.2), Fake scores high (0.8 to 1.0)
        labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        scores = np.array([0.1, 0.15, 0.05, 0.2, 0.85, 0.9, 0.8, 0.95])
        eer, thresh = compute_eer(labels, scores, score_orientation="higher_is_fake")
        self.assertEqual(eer, 0.0)
        self.assertTrue(0.2 <= thresh <= 0.85)

    def test_compute_eer_higher_is_bonafide(self):
        # Inverted scores: Genuine scores high (10.0), Fake scores low (-10.0)
        labels = np.array([0, 0, 1, 1])
        scores = np.array([10.0, 9.5, -8.0, -9.0])
        eer, thresh = compute_eer(labels, scores, score_orientation="higher_is_bonafide")
        self.assertEqual(eer, 0.0)
        self.assertTrue(-8.0 <= thresh <= 9.5)

    def test_evaluate_predictions_metrics(self):
        labels = np.array([0, 0, 1, 1])
        scores = np.array([0.1, 0.2, 0.8, 0.9])
        res = evaluate_predictions(labels, scores, threshold=0.5, score_orientation="higher_is_fake")

        self.assertEqual(res["accuracy"], 1.0)
        self.assertEqual(res["precision"], 1.0)
        self.assertEqual(res["recall"], 1.0)
        self.assertEqual(res["f1"], 1.0)
        self.assertEqual(res["roc_auc"], 1.0)
        self.assertEqual(res["genuine_false_positive_rate"], 0.0)
        self.assertEqual(res["confusion_matrix"]["true_positives"], 2)
        self.assertEqual(res["confusion_matrix"]["true_negatives"], 2)

    def test_per_attack_metrics_breakdown(self):
        rows = [
            {"label": 0, "attack_type": "-", "dataset": "asv", "split": "dev"},
            {"label": 0, "attack_type": "-", "dataset": "asv", "split": "dev"},
            {"label": 1, "attack_type": "A01", "dataset": "asv", "split": "dev"},
            {"label": 1, "attack_type": "A01", "dataset": "asv", "split": "dev"},
            {"label": 1, "attack_type": "A02", "dataset": "asv", "split": "dev"},
        ]
        scores = np.array([0.1, 0.2, 0.9, 0.85, 0.3])  # A02 missed with thresh=0.5
        attack_metrics = compute_per_attack_metrics(rows, scores, threshold=0.5)

        self.assertEqual(len(attack_metrics), 3)  # bonafide, A01, A02
        bon_row = next(r for r in attack_metrics if r["attack_type"] == "bonafide (-)")
        self.assertEqual(bon_row["total_samples"], 2)
        self.assertEqual(bon_row["false_positive_rate"], 0.0)

        a01_row = next(r for r in attack_metrics if r["attack_type"] == "A01")
        self.assertEqual(a01_row["total_samples"], 2)
        self.assertEqual(a01_row["detected_spoofs"], 2)
        self.assertEqual(a01_row["recall"], 1.0)

        a02_row = next(r for r in attack_metrics if r["attack_type"] == "A02")
        self.assertEqual(a02_row["total_samples"], 1)
        self.assertEqual(a02_row["detected_spoofs"], 0)
        self.assertEqual(a02_row["recall"], 0.0)


if __name__ == "__main__":
    unittest.main()
