"""Unit tests for command line interfaces and utilities."""
import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from summarize_results import fmt, render_categories, render_metrics
from analyze_video import parse_args


class TestCLI(unittest.TestCase):
    """Test suite for command line tools and reporting scripts."""

    def test_fmt_helper(self):
        self.assertEqual(fmt(None), "-")
        self.assertEqual(fmt(0.123456), "0.1235")
        self.assertEqual(fmt("raw_text"), "raw_text")

    def test_render_metrics(self):
        mock_results = {
            "metrics": {
                "video": {"accuracy": 0.99, "auc": 0.999, "f1": 0.995, "recall": 0.99},
                "audio": {"accuracy": 0.98, "auc": 0.998, "f1": 0.987, "recall": 0.98},
                "any": {"accuracy": 0.995, "auc": 0.9995, "f1": 0.998, "recall": 0.997},
            }
        }
        table = render_metrics(mock_results)
        self.assertIn("| video | 0.9900 | 0.9990 | 0.9950 | 0.9900 |", table)
        self.assertIn("| audio |", table)
        self.assertIn("| any |", table)

    def test_render_categories(self):
        mock_results = {
            "per_category": [
                {"category": "F-R", "n": 1485, "any_acc": 0.997, "any_recall": 0.997, "any_f1": 0.998},
            ]
        }
        table = render_categories(mock_results)
        self.assertIn("| F-R | 1485 | 0.9970 | 0.9970 | 0.9980 |", table)

    def test_analyze_video_parse_args(self):
        sys_argv_backup = sys.argv
        try:
            sys.argv = ["analyze_video.py", "--video", "test.mp4", "--json"]
            args = parse_args()
            self.assertEqual(args.video, "test.mp4")
            self.assertTrue(args.json)
            self.assertEqual(args.threshold, 0.5)
            self.assertEqual(args.model, "models/best_model.pt")
        finally:
            sys.argv = sys_argv_backup


if __name__ == "__main__":
    unittest.main()

