"""Unit and integration tests for Flask web UI endpoints."""
import io
import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src", ROOT / "ui"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from ui.app import app, safe_child


class TestUI(unittest.TestCase):
    """Test suite for Web UI routes, security guards, and API responses."""

    def setUp(self):
        app.config["TESTING"] = True
        self.client = app.test_client()

    def test_index_route(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Deepfake Detection Studio", response.data)

    def test_status_endpoint(self):
        response = self.client.get("/api/status")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["status"], "online")
        self.assertIn("multimodal_model", data)
        self.assertIn("audio_model", data)

    def test_safe_child_security(self):
        base = (ROOT / "ui" / "samples").resolve()
        # Normal safe child
        self.assertIsNotNone(safe_child(base, "sample.mp4"))
        # Traversal attacks should return None
        self.assertIsNone(safe_child(base, "../../../windows/system32/cmd.exe"))
        self.assertIsNone(safe_child(base, "../../app.py"))

    def test_serve_sample_not_found(self):
        response = self.client.get("/samples/does_not_exist.mp4")
        self.assertEqual(response.status_code, 404)

    def test_serve_upload_not_found(self):
        response = self.client.get("/uploads/does_not_exist.mp4")
        self.assertEqual(response.status_code, 404)

    def test_analyze_invalid_mode(self):
        response = self.client.post("/api/analyze", data={"mode": "magic"})
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertFalse(data["ok"])
        self.assertIn("Unknown mode", data["error"])

    def test_analyze_upload_missing_file(self):
        response = self.client.post("/api/analyze", data={"mode": "upload"})
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertFalse(data["ok"])
        self.assertIn("No video file was uploaded", data["error"])

    def test_analyze_audio_missing_file(self):
        response = self.client.post("/api/analyze-audio")
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertFalse(data["ok"])
        self.assertIn("select an audio file", data["error"])

    def test_analyze_audio_invalid_extension(self):
        data = {"file": (io.BytesIO(b"fake data"), "malicious.exe")}
        response = self.client.post("/api/analyze-audio", data=data, content_type="multipart/form-data")
        self.assertEqual(response.status_code, 400)
        json_data = response.get_json()
        self.assertFalse(json_data["ok"])
        self.assertIn("Supported audio formats", json_data["error"])


if __name__ == "__main__":
    unittest.main()

