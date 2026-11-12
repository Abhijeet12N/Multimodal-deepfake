"""Environment and hardware auditing script for deepfake detection models.

Inspects Python runtime, PyTorch CUDA capabilities, system RAM, CPU/GPU,
audio library availability, and model checkpoint feasibility.
Can output a formatted Markdown report or JSON structure.

Usage:
    python scripts/audit_environment.py [--output docs/environment_audit.md]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

# Ensure console supports utf-8
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
from deepfake_detector.upstream import default_aasist_checkpoint


def get_sha256(path: Path) -> str:
    if not path.is_file():
        return "NOT_FOUND"
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def audit_system() -> Dict[str, Any]:
    info: Dict[str, Any] = {}

    # OS & Python
    info["os"] = f"{platform.system()} {platform.release()} ({platform.version()})"
    info["machine"] = platform.machine()
    info["processor"] = platform.processor()
    info["python_version"] = sys.version
    info["is_colab"] = "google.colab" in sys.modules or os.path.exists("/content")

    # Memory & Disk
    try:
        import psutil
        vm = psutil.virtual_memory()
        info["total_ram_gb"] = round(vm.total / (1024**3), 2)
        info["available_ram_gb"] = round(vm.available / (1024**3), 2)
        info["cpu_count_logical"] = psutil.cpu_count(logical=True)
        info["cpu_count_physical"] = psutil.cpu_count(logical=False)
    except ImportError:
        info["total_ram_gb"] = "psutil not installed"
        info["available_ram_gb"] = "unknown"
        info["cpu_count_logical"] = os.cpu_count()
        info["cpu_count_physical"] = "unknown"

    disk = shutil.disk_usage(REPO_ROOT)
    info["disk_total_gb"] = round(disk.total / (1024**3), 2)
    info["disk_free_gb"] = round(disk.free / (1024**3), 2)

    # PyTorch & CUDA
    try:
        import torch
        info["torch_version"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["cuda_version"] = torch.version.cuda
            info["device_count"] = torch.cuda.device_count()
            info["device_name"] = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            info["vram_gb"] = round(vram, 2)
        else:
            info["cuda_version"] = None
            info["device_name"] = "CPU"
            info["vram_gb"] = 0.0
    except ImportError as e:
        info["torch_version"] = f"Error: {e}"
        info["cuda_available"] = False

    # Dependencies
    deps = {}
    for pkg in [
        "torchaudio",
        "transformers",
        "soundfile",
        "librosa",
        "scipy",
        "sklearn",
        "timm",
        "cv2",
        "mamba_ssm",
    ]:
        try:
            m = __import__(pkg)
            ver = getattr(m, "__version__", "installed")
            deps[pkg] = ver
        except ImportError:
            deps[pkg] = "not installed"
    info["dependencies"] = deps

    # Model Checkpoints
    ckpts = {
        "legacy_wav2vec_best": REPO_ROOT.parent / "initial setup" / "deepfake_phase1" / "checkpoints" / "wav2vec_best.pt",
        "legacy_audio": REPO_ROOT / "models" / "weights" / "audio_model.pt",
        "multimodal": REPO_ROOT / "models" / "weights" / "best_model.pt",
        "aasist": default_aasist_checkpoint(REPO_ROOT, "aasist"),
        "aasist_l": default_aasist_checkpoint(REPO_ROOT, "aasist_l"),
        "ssl_aasist_la": REPO_ROOT / "models" / "weights" / "LA_model.pth",
        "ssl_aasist_df": REPO_ROOT / "models" / "weights" / "DF_model.pth",
    }

    ckpt_info = {}
    for name, p in ckpts.items():
        if p.is_file():
            ckpt_info[name] = {
                "exists": True,
                "path": str(p),
                "size_mb": round(p.stat().st_size / (1024**2), 2),
                "sha256": get_sha256(p)[:16] + "...",
            }
        else:
            ckpt_info[name] = {
                "exists": False,
                "path": str(p),
                "size_mb": 0,
                "sha256": "N/A",
            }
    info["checkpoints"] = ckpt_info

    # Model Feasibility Assessment
    info["feasibility"] = {
        "legacy_wav2vec_mlp": {
            "status": "Ready",
            "notes": "Runs on CPU/GPU with PyTorch and Transformers.",
        },
        "aasist": {
            "status": "Ready" if ckpt_info["aasist"]["exists"] else "Weights missing",
            "notes": "PyTorch graph neural network, runs on CPU or GPU.",
        },
        "ssl_aasist": {
            "status": "Colab GPU Only" if not info["is_colab"] else "Requires fairseq",
            "notes": "Requires XLSR frontend and fairseq, heavy compute requirement.",
        },
        "xlsr_mambo": {
            "status": "Optional / Blocked",
            "notes": "Requires CUDA 12.8+, FlashAttention, and mamba-ssm.",
        },
    }

    return info


def generate_markdown(info: Dict[str, Any]) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    lines = [
        "# Environment and Hardware Audit",
        f"**Generated:** {now}",
        "",
        "## 1. System & Runtime Platform",
        f"- **Operating System:** {info['os']}",
        f"- **Architecture:** {info['machine']}",
        f"- **Python Version:** {info['python_version'].split()[0]}",
        f"- **Execution Environment:** {'Google Colab' if info['is_colab'] else 'Local Workstation'}",
        f"- **Logical CPU Cores:** {info['cpu_count_logical']}",
        f"- **System RAM:** {info['total_ram_gb']} GB (Available: {info['available_ram_gb']} GB)",
        f"- **Disk Storage:** {info['disk_free_gb']} GB free of {info['disk_total_gb']} GB total",
        "",
        "## 2. Compute Acceleration (PyTorch & CUDA)",
        f"- **PyTorch Version:** {info['torch_version']}",
        f"- **CUDA Available:** {info['cuda_available']}",
        f"- **CUDA Version:** {info['cuda_version'] or 'N/A'}",
        f"- **Device:** {info['device_name']}",
        f"- **VRAM:** {info['vram_gb']} GB",
        "",
        "## 3. Python Package Dependencies",
        "| Package | Status / Version |",
        "|---|---|",
    ]
    for pkg, ver in info["dependencies"].items():
        lines.append(f"| `{pkg}` | {ver} |")

    lines.extend([
        "",
        "## 4. Checkpoint Inventory",
        "| Identifier | Exists | Size (MB) | SHA-256 (prefix) | Path |",
        "|---|---|---|---|---|",
    ])
    for name, c in info["checkpoints"].items():
        exists_str = "[YES]" if c["exists"] else "[NO]"
        lines.append(f"| `{name}` | {exists_str} | {c['size_mb']} | `{c['sha256']}` | `{c['path']}` |")

    lines.extend([
        "",
        "## 5. Model Feasibility Matrix",
        "| Model | Feasibility Status | Operating Constraints |",
        "|---|---|---|",
    ])
    for m, d in info["feasibility"].items():
        lines.append(f"| `{m}` | **{d['status']}** | {d['notes']} |")

    lines.append("")
    return "\n".join(lines)


def main():
    if sys.stdout and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Audit environment and hardware.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write markdown audit report.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print JSON output instead of Markdown.",
    )
    args = parser.parse_args()

    info = audit_system()

    if args.json:
        out_text = json.dumps(info, indent=2)
        print(out_text)
    else:
        out_text = generate_markdown(info)
        print(out_text)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(out_text)
        print(f"\n[Audit report written to {args.output}]")


if __name__ == "__main__":
    main()
