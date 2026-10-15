"""Backward-compatibility shim. Please use training/train_audio.py."""
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
target = ROOT / "training" / "train_audio.py"

if __name__ == "__main__":
    print(f"[Notice] notebook_v2/train_audio_only.py is deprecated. Delegating to 'training/train_audio.py'.")
    runpy.run_path(str(target), run_name="__main__")
else:
    if str(ROOT / "training") not in sys.path:
        sys.path.insert(0, str(ROOT / "training"))
    from train_audio import *
