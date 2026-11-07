"""CLI entry point for audio classifier training and feature extraction.

Adapts and exposes training/train_audio.py under the scripts/ directory
as requested by the model upgradation plan.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.train_audio import main

if __name__ == "__main__":
    main()
