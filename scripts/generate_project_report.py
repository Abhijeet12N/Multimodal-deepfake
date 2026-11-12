"""Compatibility entry point for the current PPT generation report.

Use scripts/build_presentation_report.py directly for new workflows.
"""
from pathlib import Path
import runpy


def build() -> None:
    runpy.run_path(str(Path(__file__).with_name("build_presentation_report.py")), run_name="__main__")


if __name__ == "__main__":
    build()
