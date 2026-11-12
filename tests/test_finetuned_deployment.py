"""Fine-tuned checkpoint selection, provenance, and UI status regression checks."""
from pathlib import Path

import torch

import ui.app as studio
from deepfake_detector.audio_detection.detectors import aasist


def test_installed_checkpoint_is_preferred_and_pretrained_remains_fallback(tmp_path):
    root = tmp_path / "project"
    local = root / "models" / "aasist_finetuned.pth"
    upstream = root / "third_party" / "aasist" / "models" / "weights" / "AASIST.pth"
    assert studio.resolve_aasist_path(root, "aasist") == upstream
    local.parent.mkdir(parents=True)
    local.write_bytes(b"local checkpoint presence fixture")
    assert studio.resolve_aasist_path(root, "aasist") == local
    assert studio.resolve_aasist_path(root, "aasist_l").name == "AASIST-L.pth"


def test_explicit_checkpoint_overrides_installed_model(tmp_path):
    assert studio.resolve_aasist_path(tmp_path, "aasist", "custom/model.pth") == tmp_path / "custom/model.pth"
    absolute = (tmp_path / "elsewhere.pth").resolve()
    assert studio.resolve_aasist_path(tmp_path, "aasist", str(absolute)) == absolute


def test_checkpoint_provenance_survives_loading_and_resets_for_baseline(tmp_path, monkeypatch):
    model = torch.nn.Linear(2, 2)
    monkeypatch.setattr(aasist, "_load_upstream_aasist_model", lambda config, device: torch.nn.Linear(2, 2).to(device))
    fine = tmp_path / "aasist_finetuned.pth"
    torch.save({"model_state_dict": model.state_dict(), "model_id": "aasist", "epoch": 4,
                "val_auc": 0.9996, "parent_checkpoint_sha256": "parent", "manifest_sha256": "manifest"}, fine)
    detector = aasist.AASISTDetector(fine, device="cpu")
    assert detector.metadata()["training_metadata"]["epoch"] == 4
    assert "Fine-tuned AASIST" in detector.metadata()["display_name"]
    baseline = tmp_path / "AASIST.pth"
    torch.save(model.state_dict(), baseline)
    detector.load(baseline)
    assert detector.metadata()["training_metadata"] == {}
    assert "Official AASIST" in detector.metadata()["display_name"]


def test_status_and_dashboard_identify_selected_checkpoint(tmp_path, monkeypatch):
    checkpoint = tmp_path / "aasist_finetuned.pth"
    checkpoint.write_bytes(b"presence fixture; inference is mocked")
    monkeypatch.setitem(studio.AASIST_PATHS, "aasist", str(checkpoint))
    monkeypatch.setattr(studio, "DEFAULT_AUDIO_DETECTOR", "aasist")
    monkeypatch.setattr(studio, "audio_predictor_id", "aasist")
    monkeypatch.setattr(studio, "load_predictor", lambda: None)
    monkeypatch.setattr(studio, "load_audio_predictor", lambda detector_id: object())
    client = studio.app.test_client()
    status = client.get("/api/status").get_json()
    assert status["audio_model"]["path"] == str(checkpoint)
    assert status["audio_model"]["model_id"] == "aasist"
    assert status["audio_model"]["loaded"]
    dashboard = client.get("/")
    assert dashboard.status_code == 200
    assert b'AASIST (fine-tuned)' in dashboard.data
    assert b'value="aasist" selected' in dashboard.data
