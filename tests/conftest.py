"""Pytest configuration and environment fixtures for test suite."""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

for p in (REPO_ROOT, REPO_ROOT / "src", REPO_ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
