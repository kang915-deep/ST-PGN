"""C-MAPSS RUL dataset loader for ST-PGN.

The loader keeps engine units isolated when creating train/validation splits,
fits normalization statistics on the training engines only, and returns
windows with shape [T, C] plus a scalar RUL target.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


SENSOR_COUNT = 21
RAW_COLUMN_COUNT = 2 + 3 + SENSOR_COUNT


@dataclass
class CMapssStats:
    mean: np.ndarray
    std: np.ndarray


def _read_cmapss_file(path: Path) -> pd.DataFrame:
    """Read a whitespace-delimited C-MAPSS train/test file."""
    if not path.exists():
        raise FileNotFoundError(f"C-MAPSS file not found: {path}")
    df = pd.read_csv(path, sep=r"\s+", header=None, engine="python")
    # Some copies contain an extra empty trailing field; remove it safely.
    df = df.dropna(axis=1, how="all")
    if df.shape[1] != RAW_COLUMN_COUNT:
        raise ValueError(
            f"Expected {RAW_COLUMN_COUNT} columns in {path}, got {df.shape[1]}"
        )
    columns = ["unit", "cycle"] + [f"op_{i}" for i in range(1, 4)]
    columns += [f"sensor_{i}" for i in range(1, SENSOR_COUNT + 1)]
    df.columns = columns
    df["unit"] = df["unit"].astype(int)
    df["cycle"] = df["cycle"].astype(int)
    return df


def _sensor_columns(sensor_indices: Optional[Sequence[int]]) -> List[str]:
    if sensor_indices is None:
        sensor_indices = range(1, SENSOR_COUNT + 1)
    sensor_indices = list(sensor_indices)
    if not sensor_indices or any(i < 1 or i > SENSOR_COUNT for i in sensor_indices):
        raise ValueError("sensor_indices must contain values from 1 to 21")
    return [f"sensor_{i}" for i in sensor_indices]


def _fit_stats(df: pd.DataFrame, columns: Sequence[str]) -> CMapssStats:
    values = df.loc[:, columns].to_numpy(dtype=np.float32)
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    std[std < 1e-6] = 1.0
    return CMapssStats(mean=mean, std=std)


def _make_train_rul(df: pd.DataFrame, rul_max: float) -> pd.Series:
    max_cycle = df.groupby("unit")["cycle"].transform("max")
    return (max_cycle - df["cycle"]).clip(upper=rul_max).astype(np.float32)


def _make_test_rul(df: pd.DataFrame, rul_file: Path, rul_max: float) -> pd.Series:
    rul = np.loadtxt(rul_file, dtype=np.float32).reshape(-1)
    units = sorted(df["unit"].unique())
    if len(rul) != len(units):
        raise ValueError(
            f"{rul_file} contains {len(rul)} RUL values, but test data has "
            f"{len(units)} engines"
        )
    remaining = dict(zip(units, rul.tolist()))
    last_cycle = df.groupby("unit")["cycle"].transform("max")
    base_rul = df["unit"].map(remaining).to_numpy(dtype=np.float32)
    labels = base_rul + last_cycle.to_numpy(dtype=np.float32) - df["cycle"].to_numpy(dtype=np.float32)
    return pd.Series(np.minimum(labels, rul_max), index=df.index, dtype=np.float32)


class CMapssDataset(Dataset):
    """Windowed C-MAPSS dataset.

    Args:
        root: Directory containing train_FDxxx.txt/test_FDxxx.txt files.
        subset: One of FD001, FD002, FD003, FD004.
        split: train, val, or test. Validation engines are held out from the
            official training file; test uses the official test file.
        window_size: Number of cycles returned per sample.
        rul_max: Piecewise-linear RUL cap.
        stats: Training statistics. Required for val/test when reproducibility
            matters; build_cmapss_datasets supplies them automatically.
        val_ratio: Fraction of training engines held out for validation.
        sensor_indices: 1-based sensor IDs. None keeps all 21 sensors.
    """

    def __init__(
        self,
        root: str,
        subset: str = "FD001",
        split: str = "train",
        window_size: int = 30,
        rul_max: float = 125.0,
        stats: Optional[CMapssStats] = None,
        val_ratio: float = 0.2,
        seed: int = 42,
        sensor_indices: Optional[Sequence[int]] = None,
        test_last_only: bool = True,
        last_only: bool = False,
    ):
        super().__init__()
        subset = subset.upper()
        if subset not in {"FD001", "FD002", "FD003", "FD004"}:
            raise ValueError("subset must be FD001, FD002, FD003, or FD004")
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be train, val, or test")
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        if not 0.0 <= val_ratio < 1.0:
            raise ValueError("val_ratio must be in [0, 1)")

        self.root = Path(root)
        self.subset = subset
        self.split = split
        self.window_size = window_size
        self.rul_max = rul_max
        self.sensor_cols = _sensor_columns(sensor_indices)
        self.test_last_only = test_last_only
        self.last_only = last_only

        train_path = self.root / f"train_{subset}.txt"
        test_path = self.root / f"test_{subset}.txt"
        rul_path = self.root / f"RUL_{subset}.txt"
        train_df = _read_cmapss_file(train_path)

        train_units = sorted(train_df["unit"].unique().tolist())
        rng = np.random.default_rng(seed)
        shuffled = np.asarray(train_units, dtype=np.int64)
        rng.shuffle(shuffled)
        val_count = max(1, int(round(len(train_units) * val_ratio))) if val_ratio else 0
        val_units = set(shuffled[:val_count].tolist())
        fit_units = set(train_units) - val_units

        if split in {"train", "val"}:
            selected_units = fit_units if split == "train" else val_units
            df = train_df[train_df["unit"].isin(selected_units)].copy()
            df["rul"] = _make_train_rul(df, rul_max)
            if stats is None:
                stats = _fit_stats(df, self.sensor_cols)
        else:
            df = _read_cmapss_file(test_path)
            df["rul"] = _make_test_rul(df, rul_path, rul_max)
        if stats is None:
            raise ValueError("stats must be supplied for validation/test datasets")
        self.stats = stats
        self.samples: List[Tuple[np.ndarray, float]] = []
        self.sample_units: List[int] = []
        self.sample_cycles: List[int] = []

        for unit in sorted(df["unit"].unique()):
            unit_df = df[df["unit"] == unit].sort_values("cycle")
            features = unit_df.loc[:, self.sensor_cols].to_numpy(dtype=np.float32)
            features = (features - stats.mean) / stats.std
            labels = unit_df["rul"].to_numpy(dtype=np.float32)
            if len(unit_df) < window_size:
                continue
            use_last_only = (split == "test" and test_last_only) or (
                split == "val" and last_only
            )
            end_positions = [len(unit_df) - 1] if use_last_only else range(
                window_size - 1, len(unit_df)
            )
            for end in end_positions:
                start = end - window_size + 1
                self.samples.append((features[start:end + 1], float(labels[end])))
                self.sample_units.append(int(unit))
                self.sample_cycles.append(int(unit_df.iloc[end]["cycle"]))

        if not self.samples:
            raise ValueError(
                f"No windows generated for {subset}/{split}; "
                f"check window_size={window_size} and data path {self.root}"
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        x, y = self.samples[index]
        return torch.from_numpy(x), torch.tensor([y], dtype=torch.float32)


def build_cmapss_datasets(
    root: str,
    subset: str = "FD001",
    window_size: int = 30,
    rul_max: float = 125.0,
    val_ratio: float = 0.2,
    seed: int = 42,
    sensor_indices: Optional[Sequence[int]] = None,
    test_last_only: bool = True,
    val_last_only: bool = False,
) -> Dict[str, CMapssDataset]:
    """Build train/validation/test datasets sharing train-only statistics."""
    train = CMapssDataset(
        root, subset, "train", window_size, rul_max,
        val_ratio=val_ratio, seed=seed, sensor_indices=sensor_indices,
    )
    val = CMapssDataset(
        root, subset, "val", window_size, rul_max, stats=train.stats,
        val_ratio=val_ratio, seed=seed, sensor_indices=sensor_indices,
        last_only=val_last_only,
    )
    test = CMapssDataset(
        root, subset, "test", window_size, rul_max, stats=train.stats,
        val_ratio=val_ratio, seed=seed, sensor_indices=sensor_indices,
        test_last_only=test_last_only,
    )
    return {"train": train, "val": val, "test": test}
