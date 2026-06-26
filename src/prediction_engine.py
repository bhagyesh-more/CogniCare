"""
prediction_engine.py
Loads persisted model artifacts and provides a clean inference API.

Usage
-----
    engine = PredictionEngine(models_dir="models").load()

    # Single sample as dict
    result = engine.predict_emotional_arousal({
        "eda_mean": 0.42, "eda_std": 0.07, ...
    })

    # Batch as DataFrame
    results = engine.predict_cognitive_load(df)
"""

import json
import logging
import pickle
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

FEATURE_COLS = [
    "eda_mean", "eda_std", "eda_peak_count", "eda_peak_amplitude",
    "heart_rate_bpm", "rmssd", "sdnn",
    "resp_rate_bpm", "resp_variability",
]

# Artifact sub-directories created by ModelTrainer
_TARGET_DIRS = {
    "arousal":        "arousal",
    "cognitive_load": "cognitive_load",
}


class PredictionEngine:
    """
    Reusable inference engine for emotional arousal and cognitive load.

    Loads model.pkl + scaler.pkl for each target; exposes typed
    predict_* methods that accept either a dict (single sample)
    or a DataFrame (batch).
    """

    def __init__(self, models_dir: Union[str, Path] = "models"):
        self.models_dir = Path(models_dir)
        self._artifacts: dict[str, dict] = {}   # target → {model, le, scaler, metadata}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self) -> "PredictionEngine":
        """Load all persisted artifacts from disk."""
        for target, subdir in _TARGET_DIRS.items():
            artifact_dir = self.models_dir / subdir
            self._artifacts[target] = self._load_target(target, artifact_dir)
            logger.info("PredictionEngine: loaded '%s' model.", target)
        return self

    # ------------------------------------------------------------------
    # Public inference API
    # ------------------------------------------------------------------

    def predict_emotional_arousal(
        self,
        data: Union[dict, pd.DataFrame],
    ) -> pd.DataFrame:
        """
        Predict emotional arousal level (Low / Medium / High).

        Parameters
        ----------
        data : dict (single sample) or DataFrame (batch).
               Must contain all 9 physiological feature columns.

        Returns
        -------
        DataFrame with columns: predicted_class, confidence, [probabilities per class]
        """
        return self._predict("arousal", data)

    def predict_cognitive_load(
        self,
        data: Union[dict, pd.DataFrame],
    ) -> pd.DataFrame:
        """
        Predict cognitive load level (Low / Medium / High).

        Parameters
        ----------
        data : dict (single sample) or DataFrame (batch).

        Returns
        -------
        DataFrame with columns: predicted_class, confidence, [probabilities per class]
        """
        return self._predict("cognitive_load", data)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _predict(self, target: str, data: Union[dict, pd.DataFrame]) -> pd.DataFrame:
        if target not in self._artifacts:
            raise RuntimeError(
                f"Model for '{target}' not loaded. Call load() first."
            )

        X = self._to_feature_matrix(data)
        art = self._artifacts[target]

        X_scaled   = art["scaler"].transform(X)
        y_pred_enc = art["model"].predict(X_scaled)
        y_proba    = art["model"].predict_proba(X_scaled)

        classes     = art["label_encoder"].classes_
        predictions = art["label_encoder"].inverse_transform(y_pred_enc)
        confidence  = y_proba.max(axis=1)

        result = pd.DataFrame({"predicted_class": predictions, "confidence": confidence})
        for i, cls in enumerate(classes):
            result[f"prob_{cls}"] = y_proba[:, i]

        return result

    @staticmethod
    def _to_feature_matrix(data: Union[dict, pd.DataFrame]) -> pd.DataFrame:
        """Normalise input to a DataFrame with exactly FEATURE_COLS columns."""
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        else:
            df = data.copy()

        missing = [c for c in FEATURE_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"Input is missing required features: {missing}")

        # Impute NaN with 0 (scaler was fit on clean data; edge case only)
        return df[FEATURE_COLS].fillna(0.0)

    @staticmethod
    def _load_target(target: str, artifact_dir: Path) -> dict:
        """Load model.pkl, scaler.pkl and feature_metadata.json for one target."""
        model_path  = artifact_dir / "model.pkl"
        scaler_path = artifact_dir / "scaler.pkl"
        meta_path   = artifact_dir / "feature_metadata.json"

        for p in (model_path, scaler_path):
            if not p.exists():
                raise FileNotFoundError(
                    f"Artifact not found: {p}\n"
                    "Run train.py first to generate model artifacts."
                )

        with open(model_path, "rb") as f:
            bundle = pickle.load(f)

        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)

        metadata = {}
        if meta_path.exists():
            with open(meta_path) as f:
                metadata = json.load(f)

        return {
            "model":          bundle["model"],
            "label_encoder":  bundle["label_encoder"],
            "scaler":         scaler,
            "metadata":       metadata,
        }
