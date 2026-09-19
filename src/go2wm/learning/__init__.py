"""Person B / ML pipeline: dataset consumer, splits, readouts, evaluation, bundles.

This package needs NumPy (``uv sync --extra ml``).  The core contracts,
planner, and runtime packages stay dependency-free.
"""

from .dataset import DatasetLoadError, LoadedDataset, load_dataset, load_episode
from .diagnostics import classify_block, coverage_report
from .splits import (
    LeakReport,
    SplitFractions,
    SplitLeakError,
    SplitManifest,
    build_split_manifest,
    check_leakage,
    read_split_manifest,
    write_split_manifest,
)
from .windows import PredictionWindow, dataset_windows, episode_windows, labeled_observations

__all__ = [
    "DatasetLoadError",
    "LeakReport",
    "LoadedDataset",
    "PredictionWindow",
    "SplitFractions",
    "SplitLeakError",
    "SplitManifest",
    "build_split_manifest",
    "check_leakage",
    "classify_block",
    "coverage_report",
    "dataset_windows",
    "episode_windows",
    "labeled_observations",
    "load_dataset",
    "load_episode",
    "read_split_manifest",
    "write_split_manifest",
]
