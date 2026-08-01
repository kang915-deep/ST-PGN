"""N-CMAPSS (New CMAPSS) dataset loader for ST-PGN.

N-CMAPSS provides high-fidelity turbofan degradation data with realistic
flight profiles.  Data is distributed as HDF5 files from NASA's Prognostics
Data Repository.

Expected directory layout::

    root/
        N-CMAPSS_DS01-005.h5
        N-CMAPSS_DS02-006.h5
        ...

Each HDF5 file contains these datasets:
    W   — operating conditions (4 columns: altitude, Mach, TRA, T2)
    X_s — sensor measurements (~14 columns)
    X_v — virtual sensors (~4 columns, optional)
    T   — temperature/pressure (~4 columns, optional)
    Y   — RUL labels (1 column, cycles to failure)
    A   — auxiliary info (unit ID + cycle columns)
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import h5py
except ImportError:
    h5py = None  # type: ignore[assignment]


@dataclass
class NCMapssStats:
    mean: np.ndarray
    std: np.ndarray


def _require_h5py():
    if h5py is None:
        raise ImportError(
            "h5py is required for N-CMAPSS datasets.  "
            "Install it with:  pip install h5py"
        )


def _load_h5_dataset(filepath: Path) -> Dict[str, np.ndarray]:
    """Load all arrays from a single N-CMAPSS HDF5 file."""
    _require_h5py()
    if not filepath.exists():
        raise FileNotFoundError(f"N-CMAPSS file not found: {filepath}")

    data: Dict[str, np.ndarray] = {}
    with h5py.File(filepath, "r") as f:
        # Iterate through possible dataset keys
        for key in ["W", "X_s", "X_v", "T", "Y", "A"]:
            for variant in [f"/{key}_dev", f"/{key}_test", f"/{key}"]:
                if variant in f:
                    arr = np.array(f[variant], dtype=np.float32)
                    base_key = key + ("_dev" if "_dev" in variant else
                                      "_test" if "_test" in variant else "")
                    data[base_key] = arr
    return data


def _discover_h5_files(root: Path, subset: str) -> List[Path]:
    """Find HDF5 files matching the given subset pattern."""
    pattern = f"*{subset}*"
    files = sorted(root.glob(pattern))
    h5_files = [f for f in files if f.suffix in {".h5", ".hdf5"}]
    if not h5_files:
        raise FileNotFoundError(
            f"No N-CMAPSS HDF5 files matching '{pattern}' found in {root}\n"
            f"Expected files like: N-CMAPSS_DS01-005.h5"
        )
    return h5_files


def _fit_stats(features: np.ndarray) -> NCMapssStats:
    """Compute mean/std from training features."""
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    std[std < 1e-6] = 1.0
    return NCMapssStats(mean=mean, std=std)


class NCMapssDataset(Dataset):
    """Windowed N-CMAPSS dataset.

    Args:
        root: Directory containing N-CMAPSS HDF5 files.
        subset: Dataset subset identifier (e.g. "DS01", "DS02").
        split: One of "train", "val", "test".
        window_size: Number of time steps per sample window.
        rul_max: Piecewise-linear RUL cap.
        stats: Normalization statistics from training set.
        val_ratio: Fraction of training units held out for validation.
        seed: Random seed for reproducible train/val splits.
        stride: Stride for sliding window extraction (default: 1).
        use_operating: Whether to include operating conditions (W).
        use_virtual: Whether to include virtual sensors (X_v).
        test_last_only: For test split, use only the final window per unit.
    """

    def __init__(
        self,
        root: str,
        subset: str = "DS01",
        split: str = "train",
        window_size: int = 30,
        rul_max: float = 125.0,
        stats: Optional[NCMapssStats] = None,
        val_ratio: float = 0.2,
        seed: int = 42,
        stride: int = 10,
        use_operating: bool = True,
        use_virtual: bool = False,
        test_last_only: bool = True,
    ):
        _require_h5py()
        super().__init__()
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be train, val, or test")
        if window_size <= 0:
            raise ValueError("window_size must be positive")

        self.root = Path(root)
        self.subset = subset
        self.split = split
        self.window_size = window_size
        self.rul_max = rul_max
        self.stride = stride

        h5_files = _discover_h5_files(self.root, subset)

        # Load and concatenate data from all matching files
        all_features_list: List[np.ndarray] = []
        all_rul_list: List[np.ndarray] = []
        all_units_list: List[np.ndarray] = []
        unit_offset = 0

        for h5_path in h5_files:
            data = _load_h5_dataset(h5_path)

            # Determine which split-specific keys are available
            suffix = "_dev" if split in {"train", "val"} else "_test"
            fallback = ""

            # Build feature columns
            feat_parts = []
            for key_base in ["X_s"]:  # sensor measurements (always)
                key = key_base + suffix if (key_base + suffix) in data else key_base
                if key in data:
                    feat_parts.append(data[key])

            if use_operating:
                key = "W" + suffix if ("W" + suffix) in data else "W"
                if key in data:
                    feat_parts.append(data[key])

            if use_virtual:
                key = "X_v" + suffix if ("X_v" + suffix) in data else "X_v"
                if key in data:
                    feat_parts.append(data[key])

            if not feat_parts:
                raise ValueError(f"No feature data found in {h5_path}")

            features = np.concatenate(feat_parts, axis=1)

            # RUL labels
            rul_key = "Y" + suffix if ("Y" + suffix) in data else "Y"
            if rul_key not in data:
                raise ValueError(f"No RUL labels found in {h5_path}")
            rul = data[rul_key].squeeze()

            # Unit IDs
            aux_key = "A" + suffix if ("A" + suffix) in data else "A"
            if aux_key in data:
                units = data[aux_key][:, 0].astype(np.int64) + unit_offset
            else:
                units = np.zeros(len(features), dtype=np.int64) + unit_offset

            unit_offset = int(units.max()) + 1

            all_features_list.append(features)
            all_rul_list.append(rul)
            all_units_list.append(units)

        all_features = np.concatenate(all_features_list, axis=0)
        all_rul = np.concatenate(all_rul_list, axis=0)
        all_units = np.concatenate(all_units_list, axis=0)

        # Cap RUL
        all_rul = np.minimum(all_rul, rul_max)

        # Train/val split by unit ID
        unique_units = np.unique(all_units)
        rng = np.random.default_rng(seed)
        shuffled = unique_units.copy()
        rng.shuffle(shuffled)

        if split in {"train", "val"}:
            val_count = max(1, int(round(len(unique_units) * val_ratio)))
            val_units = set(shuffled[:val_count].tolist())
            train_units = set(shuffled[val_count:].tolist())
            selected_units = train_units if split == "train" else val_units
        else:
            selected_units = set(unique_units.tolist())

        # Filter by selected units
        mask = np.isin(all_units, list(selected_units))
        features = all_features[mask]
        rul = all_rul[mask]
        units = all_units[mask]

        # Fit or apply statistics
        if stats is None:
            if split == "train":
                stats = _fit_stats(features)
            else:
                raise ValueError("stats must be supplied for val/test splits")
        self.stats = stats

        features = (features - stats.mean) / stats.std

        # Extract sliding windows per unit
        self.samples: List[Tuple[np.ndarray, float]] = []
        self.sample_units: List[int] = []
        self.sample_cycles: List[int] = []

        for unit_id in sorted(set(units.tolist())):
            unit_mask = units == unit_id
            unit_feats = features[unit_mask]
            unit_rul = rul[unit_mask]

            if len(unit_feats) < window_size:
                continue

            if split == "test" and test_last_only:
                end_positions = [len(unit_feats) - 1]
            else:
                end_positions = list(range(
                    window_size - 1, len(unit_feats), self.stride
                ))
                # Always include the last position
                if end_positions[-1] != len(unit_feats) - 1:
                    end_positions.append(len(unit_feats) - 1)

            for end in end_positions:
                start = end - window_size + 1
                self.samples.append((
                    unit_feats[start:end + 1].copy(),
                    float(unit_rul[end]),
                ))
                self.sample_units.append(int(unit_id))
                self.sample_cycles.append(end)

        if not self.samples:
            raise ValueError(
                f"No windows generated for {subset}/{split}; "
                f"check window_size={window_size} and data path {self.root}"
            )

    @property
    def n_features(self) -> int:
        """Number of feature channels per time step."""
        return self.samples[0][0].shape[-1]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        x, y = self.samples[index]
        return torch.from_numpy(x), torch.tensor([y], dtype=torch.float32)


def build_ncmapss_datasets(
    root: str,
    subset: str = "DS01",
    window_size: int = 50,
    rul_max: float = 125.0,
    val_ratio: float = 0.2,
    seed: int = 42,
    stride: int = 10,
    use_operating: bool = True,
    use_virtual: bool = False,
    test_last_only: bool = True,
) -> Dict[str, NCMapssDataset]:
    """Build train/validation/test datasets sharing train-only statistics."""
    train = NCMapssDataset(
        root, subset, "train", window_size, rul_max,
        val_ratio=val_ratio, seed=seed, stride=stride,
        use_operating=use_operating, use_virtual=use_virtual,
    )
    val = NCMapssDataset(
        root, subset, "val", window_size, rul_max, stats=train.stats,
        val_ratio=val_ratio, seed=seed, stride=stride,
        use_operating=use_operating, use_virtual=use_virtual,
    )
    test = NCMapssDataset(
        root, subset, "test", window_size, rul_max, stats=train.stats,
        val_ratio=val_ratio, seed=seed, stride=stride,
        use_operating=use_operating, use_virtual=use_virtual,
        test_last_only=test_last_only,
    )
    return {"train": train, "val": val, "test": test}
