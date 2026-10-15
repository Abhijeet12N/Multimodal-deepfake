"""Backward-compatibility shim. Please use training/train_multimodal.py."""
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
target = ROOT / "training" / "train_multimodal.py"

if __name__ == "__main__":
    print(f"[Notice] notebook_v2/deepfake_v3.py is deprecated. Delegating to 'training/train_multimodal.py'.")
    runpy.run_path(str(target), run_name="__main__")
else:
    if str(ROOT / "training") not in sys.path:
        sys.path.insert(0, str(ROOT / "training"))
    from train_multimodal import *
