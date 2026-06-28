"""
src/validator.py
Input validation layer for CogniArousal inference pipeline.

Validates:
    - Feature presence (all 9 required columns present)
    - Value ranges (no extreme outliers beyond physical plausibility)
    - CSV schema correctness
    - Model artifact availability

Returns structured ValidationResult so UI can render targeted error messages.
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

FEATURE_COLS = [
    "eda_mean", "eda_std", "eda_peak_count", "eda_peak_amplitude",
    "heart_rate_bpm", "rmssd", "sdnn", "resp_rate_bpm", "resp_variability",
]

# Realistic physiological ranges for validation
RAW_RANGES = {
    "eda_mean": (0.0, 20.0),
    "eda_std": (0.0, 10.0),
    "eda_peak_count": (0.0, 100.0),
    "eda_peak_amplitude": (0.0, 10.0),
    "heart_rate_bpm": (40.0, 180.0),
    "rmssd": (5.0, 250.0),
    "sdnn": (10.0, 300.0),
    "resp_rate_bpm": (5.0, 40.0),
    "resp_variability": (0.0, 20.0),
}

MODELS_DIR = Path("models")
_REQUIRED_ARTIFACTS = ["model.pkl", "scaler.pkl", "feature_metadata.json"]


@dataclass
class ValidationResult:
    valid:    bool
    errors:   list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def add_error(self, msg: str) -> None:
        self.errors.append(msg)
        self.valid = False

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)


class Validator:
    """
    Validates inference inputs and model artifact availability.
    All public methods return a ValidationResult - never raise.
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate_sample(self, data: dict) -> ValidationResult:
        """Validate a single-sample dict."""
        result = ValidationResult(valid=True)
        self._check_feature_presence_dict(data, result)
        if result.valid:
            self._check_value_ranges_dict(data, result)
        return result

    def validate_dataframe(self, df: pd.DataFrame) -> ValidationResult:
        """Validate a batch CSV DataFrame."""
        result = ValidationResult(valid=True)

        if df.empty:
            result.add_error("Uploaded file is empty - no rows found.")
            return result

        self._check_schema(df, result)
        if result.valid:
            self._check_value_ranges_df(df, result)

        return result

    def validate_models(self) -> ValidationResult:
        """Check that all required model artifacts exist on disk."""
        result = ValidationResult(valid=True)
        for target in ("arousal", "cognitive_load"):
            for artifact in _REQUIRED_ARTIFACTS:
                path = MODELS_DIR / target / artifact
                if not path.exists():
                    result.add_error(
                        f"Missing artifact: {path}  "
                        f"Run `python train.py` to generate model files."
                    )
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _check_feature_presence_dict(data: dict, result: ValidationResult) -> None:
        missing = [f for f in FEATURE_COLS if f not in data]
        if missing:
            result.add_error(
                f"Missing required features: {missing}. "
                f"All 9 physiological features must be provided."
            )
        for col in FEATURE_COLS:
            if col in data and data[col] is None:
                result.add_error(f"Feature '{col}' is None - a numeric value is required.")

    @staticmethod
    def _check_value_ranges_dict(data: dict, result: ValidationResult) -> None:
        for col in FEATURE_COLS:
            val = data.get(col)
            if val is None:
                continue
            try:
                fval = float(val)
            except (ValueError, TypeError):
                result.add_error(f"Feature '{col}' must be a numeric value.")
                continue
            if not np.isfinite(fval):
                result.add_error(f"Feature '{col}' is not finite ({val}).")
                continue

            min_v, max_v = RAW_RANGES[col]
            if not (min_v <= fval <= max_v):
                result.add_error(
                    f"Feature '{col}' = {fval:.3f} is outside the allowed physiological "
                    f"range [{min_v}, {max_v}]."
                )

    @staticmethod
    def _check_schema(df: pd.DataFrame, result: ValidationResult) -> None:
        missing = [c for c in FEATURE_COLS if c not in df.columns]
        if missing:
            result.add_error(
                f"CSV is missing required columns: {missing}. "
                f"Expected columns: {FEATURE_COLS}"
            )
            return

        non_numeric = [
            c for c in FEATURE_COLS
            if not pd.api.types.is_numeric_dtype(df[c])
        ]
        if non_numeric:
            result.add_error(
                f"Non-numeric data in columns: {non_numeric}. "
                f"All feature columns must contain float values."
            )

    @staticmethod
    def _check_value_ranges_df(df: pd.DataFrame, result: ValidationResult) -> None:
        nan_counts = df[FEATURE_COLS].isna().sum()
        bad_cols = nan_counts[nan_counts > 0]
        if not bad_cols.empty:
            result.add_warning(
                f"NaN values detected in: "
                f"{bad_cols.to_dict()}. Rows with NaN will be imputed with 0."
            )

        for col in FEATURE_COLS:
            if col not in df.columns:
                continue
            # Check for non-finite values in df
            non_finite = (~np.isfinite(df[col])).sum()
            if non_finite > 0:
                result.add_error(f"Column '{col}' contains {non_finite} non-finite value(s).")
                continue

            min_v, max_v = RAW_RANGES[col]
            out_of_range = ((df[col] < min_v) | (df[col] > max_v)).sum()
            if out_of_range > 0:
                result.add_error(
                    f"{out_of_range} row(s) in '{col}' fall outside the allowed physiological "
                    f"range [{min_v}, {max_v}]."
                )

        if len(df) > 10_000:
            result.add_warning(
                f"Large batch: {len(df)} rows. Processing may take several seconds."
            )

