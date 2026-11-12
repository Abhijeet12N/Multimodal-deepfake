"""Locate the pinned AASIST dependency without importing ML libraries."""
from pathlib import Path


AASIST_REVISION = "a04c9863f63d44471dde8a6abcb3b082b07cd1d1"


def aasist_source_candidates(project_root: Path) -> tuple[Path, ...]:
    """Prefer the submodule; retain older workspace and Colab layouts."""
    return (
        project_root / "third_party" / "aasist",
        project_root / "aasist",
        project_root.parent / "aasist",
    )


def find_aasist_source(project_root: Path) -> Path | None:
    return next((path.resolve() for path in aasist_source_candidates(project_root)
                 if (path / "models" / "AASIST.py").is_file()), None)


def aasist_checkpoint_candidates(project_root: Path, model_id: str) -> tuple[Path, ...]:
    if model_id not in {"aasist", "aasist_l"}:
        raise ValueError(f"Unsupported AASIST model: {model_id}")
    filename = "AASIST-L.pth" if model_id == "aasist_l" else "AASIST.pth"
    return (
        project_root / "models" / f"{model_id}_finetuned.pth",
        *(path / "models" / "weights" / filename for path in aasist_source_candidates(project_root)),
    )


def default_aasist_checkpoint(project_root: Path, model_id: str) -> Path:
    candidates = aasist_checkpoint_candidates(project_root, model_id)
    # If none exists, report the expected submodule baseline path to the caller.
    return next((path for path in candidates if path.is_file()), candidates[1])
