"""Backward-compatibility shim. The primary web interface has moved to ui/app.py."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from ui.app import app, main

if __name__ == "__main__":
    print("[Notice] demo_site/app.py is a backward-compatible entry point. Please use 'python ui/app.py'.")
    main()
