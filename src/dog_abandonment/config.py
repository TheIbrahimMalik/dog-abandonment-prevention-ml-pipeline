from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os


@dataclass(frozen=True)
class Paths:
    """Filesystem locations used by the pipeline.

    All paths can be overridden via environment variables to keep the code portable.
    """
    repo_root: Path = Path(__file__).resolve().parents[2]

    data_dir: Path = Path(os.getenv("DATA_DIR", "data"))
    raw_dir: Path = Path(os.getenv("RAW_DIR", "data/raw"))
    processed_dir: Path = Path(os.getenv("PROCESSED_DIR", "data/processed"))

    models_dir: Path = Path(os.getenv("MODELS_DIR", "models"))
    reports_dir: Path = Path(os.getenv("REPORTS_DIR", "reports"))
    figures_dir: Path = Path(os.getenv("FIGURES_DIR", "figures"))

    def abs(self) -> "Paths":
        root = self.repo_root
        return Paths(
            repo_root=root,
            data_dir=(root / self.data_dir).resolve(),
            raw_dir=(root / self.raw_dir).resolve(),
            processed_dir=(root / self.processed_dir).resolve(),
            models_dir=(root / self.models_dir).resolve(),
            reports_dir=(root / self.reports_dir).resolve(),
            figures_dir=(root / self.figures_dir).resolve(),
        )
