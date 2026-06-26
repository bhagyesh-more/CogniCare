"""
dataset_processor.py
Handles WESAD dataset loading, signal extraction, and label mapping.

WESAD Structure (per subject pickle):
    data['signal']['chest']:  ACC, ECG, EDA, EMG, Resp, Temp  @ 700 Hz
    data['signal']['wrist']:  ACC, BVP, EDA, TEMP             @ various Hz
    data['label']:            per-sample label array

Label mapping:
    0 = not defined, 1 = baseline, 2 = stress, 3 = amusement, 4+ = other
"""

import pickle
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# Only retain scientifically validated affect states
LABEL_MAP = {1: "baseline", 2: "stress", 3: "amusement"}

# Chest RespiBAN sampling rates (Hz)
CHEST_FS = {
    "ECG": 700,
    "EDA": 700,
    "Resp": 700,
}


class DatasetProcessor:
    """
    Loads a single WESAD subject file, validates its structure,
    extracts physiological signals, and aligns labels.
    """

    REQUIRED_CHEST_SIGNALS = {"ECG", "EDA", "Resp"}

    def __init__(self, subject_path: str | Path):
        self.subject_path = Path(subject_path)
        self.raw: dict = {}
        self.signals: dict[str, np.ndarray] = {}
        self.labels: np.ndarray = np.array([])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self) -> "DatasetProcessor":
        """Load and validate the subject pickle file."""
        if not self.subject_path.exists():
            raise FileNotFoundError(f"Subject file not found: {self.subject_path}")

        with open(self.subject_path, "rb") as f:
            self.raw = pickle.load(f, encoding="latin1")

        self._validate()
        logger.info("Loaded: %s", self.subject_path.name)
        return self

    def extract_signals(self) -> "DatasetProcessor":
        """Pull ECG, EDA, and Respiration arrays from chest sensor data."""
        chest = self.raw["signal"]["chest"]

        self.signals = {
            "ECG":  chest["ECG"].flatten().astype(np.float64),
            "EDA":  chest["EDA"].flatten().astype(np.float64),
            "Resp": chest["Resp"].flatten().astype(np.float64),
        }
        self.labels = self.raw["label"].flatten().astype(np.int8)

        self._handle_missing()
        logger.info(
            "Signals extracted - ECG: %d, EDA: %d, Resp: %d, Labels: %d",
            len(self.signals["ECG"]),
            len(self.signals["EDA"]),
            len(self.signals["Resp"]),
            len(self.labels),
        )
        return self

    def get_labeled_dataframe(self) -> pd.DataFrame:
        """
        Return a sample-level DataFrame containing raw signals + mapped labels.
        Drops samples whose labels are outside LABEL_MAP (0, 4, 5, 6, 7).
        """
        n = min(len(v) for v in self.signals.values())
        label_slice = self.labels[:n]

        df = pd.DataFrame(
            {
                "ECG":  self.signals["ECG"][:n],
                "EDA":  self.signals["EDA"][:n],
                "Resp": self.signals["Resp"][:n],
                "label_id": label_slice,
            }
        )

        df["label"] = df["label_id"].map(LABEL_MAP)
        df = df[df["label"].notna()].drop(columns="label_id").reset_index(drop=True)

        logger.info(
            "Label distribution:\n%s",
            df["label"].value_counts().to_string(),
        )
        return df

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _validate(self) -> None:
        """Assert expected keys exist in the loaded pickle."""
        try:
            chest_keys = set(self.raw["signal"]["chest"].keys())
        except KeyError as exc:
            raise ValueError(f"Unexpected WESAD file structure: {exc}") from exc

        missing = self.REQUIRED_CHEST_SIGNALS - chest_keys
        if missing:
            raise ValueError(f"Missing chest signals: {missing}")

        if "label" not in self.raw:
            raise ValueError("Label array not found in subject file.")

    def _handle_missing(self) -> None:
        """Replace NaN / Inf values with linear interpolation, then forward-fill."""
        for key, arr in self.signals.items():
            if not np.isfinite(arr).all():
                n_bad = (~np.isfinite(arr)).sum()
                logger.warning("%s: replacing %d non-finite values via interpolation.", key, n_bad)
                series = pd.Series(arr).replace([np.inf, -np.inf], np.nan)
                self.signals[key] = series.interpolate(method="linear").ffill().bfill().to_numpy()
