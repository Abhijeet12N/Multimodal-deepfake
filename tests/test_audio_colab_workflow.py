"""Check generated Colab stages connect without a dataset or outputs on Drive."""
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import zipfile


def test_generated_download_to_training_workflow_uses_local_storage(tmp_path):
    notebook_path = Path(__file__).resolve().parents[1] / "notebooks" / "AASIST_Colab_Upgrade.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    cells = ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]
    installation = next(cell for cell in cells if "%pip install" in cell)
    assert "transformers>=" not in installation
    assert "huggingface-hub" not in installation
    download_cell = next(cell for cell in cells if "DOWNLOAD_ROOT =" in cell)
    manifest_cell = next(cell for cell in cells if "results_home =" in cell)
    training_cell = next(cell for cell in cells if "for model_id in MODELS:" in cell and "finetune_aasist.py" in cell)
    export_cell = next(cell for cell in cells if "export_path =" in cell)
    project = tmp_path / "project"
    project.mkdir()
    for name in ("AASIST.pth", "AASIST-L.pth"):
        weights = tmp_path / "aasist" / "models" / "weights" / name
        weights.parent.mkdir(parents=True, exist_ok=True)
        weights.write_bytes(b"synthetic checkpoint for workflow test")

    def colab_path(value):
        value = str(value)
        if value.startswith("/content/"):
            return tmp_path / value.removeprefix("/content/")
        return Path(value)

    stages, downloads = [], []
    def run(command, check):
        assert check
        if str(command[2]).endswith("download_asvspoof2019.py"):
            stages.append("download")
            destination = Path(command[command.index("--output-dir") + 1])
            root = destination / "LA"
            protocols = root / "ASVspoof2019_LA_cm_protocols"
            protocols.mkdir(parents=True)
            for split, suffix in (("train", "train.trn"), ("dev", "dev.trl"), ("eval", "eval.trl")):
                folder = root / f"ASVspoof2019_LA_{split}" / "flac"
                folder.mkdir(parents=True)
                rows = []
                for speaker in [f"{split}_speaker_1", f"{split}_speaker_2"]:
                    for key in ("bonafide", "spoof"):
                        sample = f"{speaker}_{key}"
                        (folder / f"{sample}.flac").write_bytes(b"fLaC")
                        rows.append(f"{speaker} {sample} - A01 {key}\n")
                (protocols / f"ASVspoof2019.LA.cm.{suffix}.txt").write_text("".join(rows))
            (destination / "dataset_ready.json").write_text(json.dumps({"data_root": str(root), "splits": {}}))
        else:
            assert str(command[2]).endswith("finetune_aasist.py")
            assert Path(command[command.index("--checkpoint") + 1]).is_file()
            model_id = command[command.index("--model") + 1]
            stages.append(model_id)
            output = Path(command[command.index("--output") + 1])
            output.mkdir(parents=True)
            (output / "evaluation.json").write_text("{}")
            (output / f"{model_id}_finetuned.pth").write_bytes(b"synthetic output")

    context = {"Path": colab_path, "project": project, "workspace": tmp_path, "csv": csv,
               "datetime": datetime, "json": json, "hashlib": hashlib, "sys": sys,
               "zipfile": zipfile, "subprocess": SimpleNamespace(run=run),
               "files": SimpleNamespace(download=lambda path: downloads.append(path))}
    exec(compile(download_cell, "download_cell", "exec"), context)
    exec(compile(manifest_cell, "manifest_cell", "exec"), context)
    assert not context["SAVE_OUTPUTS_TO_DRIVE"]
    assert str(context["DATA_ROOT"]).startswith(str(tmp_path))
    with context["manifest"].open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert {row["split"] for row in rows} == {"train", "val", "dev", "eval"}
    validation = {r["speaker_id"] for r in rows if r["split"] == "val"}
    calibration = {r["speaker_id"] for r in rows if r["split"] == "dev"}
    assert not validation & calibration
    exec(compile(training_cell, "training_cell", "exec"), context)
    exec(compile(export_cell, "export_cell", "exec"), context)
    assert stages == ["download", "aasist", "aasist_l"]
    assert len(downloads) == 1 and Path(downloads[0]).is_file()
    with zipfile.ZipFile(downloads[0]) as exported:
        assert "aasist/aasist_finetuned.pth" in exported.namelist()
        assert "aasist_l/aasist_l_finetuned.pth" in exported.namelist()
