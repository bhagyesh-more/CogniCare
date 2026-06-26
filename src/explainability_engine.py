"""
explainability_engine.py
SHAP-based explainability for trained Random Forest models.

Provides:
    - Local explanations  : SHAP values for a single prediction
    - Global explanations : mean |SHAP| importances across a dataset
    - Top-N contributors  : ranked feature list with direction and magnitude
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import shap

logger = logging.getLogger(__name__)

FEATURE_COLS = [
    "eda_mean", "eda_std", "eda_peak_count", "eda_peak_amplitude",
    "heart_rate_bpm", "rmssd", "sdnn",
    "resp_rate_bpm", "resp_variability",
]

# Human-readable signal labels for narrative generation
FEATURE_LABELS: dict[str, str] = {
    "eda_mean":           "mean EDA level",
    "eda_std":            "EDA variability",
    "eda_peak_count":     "EDA peak frequency",
    "eda_peak_amplitude": "EDA peak amplitude",
    "heart_rate_bpm":     "heart rate",
    "rmssd":              "HRV (RMSSD)",
    "sdnn":               "HRV (SDNN)",
    "resp_rate_bpm":      "respiration rate",
    "resp_variability":   "respiration variability",
}


@dataclass
class FeatureContribution:
    feature:     str
    label:       str          # human-readable signal name
    shap_value:  float        # signed SHAP value for predicted class
    direction:   str          # "elevated" | "reduced"
    magnitude:   float        # abs(shap_value)


@dataclass
class LocalExplanation:
    target:           str
    predicted_class:  str
    feature_contributions: list[FeatureContribution] = field(default_factory=list)
    shap_values_raw:  np.ndarray | None = field(default=None, repr=False)


class ExplainabilityEngine:
    """
    Wraps a fitted RandomForestClassifier with a SHAP TreeExplainer.

    Parameters
    ----------
    model        : fitted RandomForestClassifier
    scaler       : fitted StandardScaler used during training
    class_names  : ordered list of class names matching model.classes_
    target       : 'arousal' or 'cognitive_load' (used in output labels)
    """

    def __init__(
        self,
        model,
        scaler,
        class_names: list[str],
        target: str,
    ):
        self.model       = model
        self.scaler      = scaler
        self.class_names = class_names
        self.target      = target
        self._explainer: shap.TreeExplainer | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def build(self, background: pd.DataFrame | None = None) -> "ExplainabilityEngine":
        """
        Initialise the SHAP TreeExplainer.

        Parameters
        ----------
        background : optional DataFrame of background samples (feature cols only).
                     If provided, used as the SHAP background distribution.
                     If None, uses the model directly (faster, no background needed
                     for TreeExplainer with tree_path_dependent).
        """
        if background is not None:
            bg_scaled = self.scaler.transform(background[FEATURE_COLS].fillna(0.0))
            self._explainer = shap.TreeExplainer(self.model, bg_scaled)
        else:
            self._explainer = shap.TreeExplainer(
                self.model, feature_perturbation="tree_path_dependent"
            )
        logger.info("ExplainabilityEngine ready for target '%s'.", self.target)
        return self

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def explain_local(
        self,
        sample: Union[dict, pd.DataFrame],
        predicted_class: str,
        top_n: int = 3,
    ) -> LocalExplanation:
        """
        Compute SHAP values for a single sample and return ranked contributions.

        Parameters
        ----------
        sample          : single-row dict or 1-row DataFrame
        predicted_class : the class predicted by the model for this sample
        top_n           : number of top contributors to return

        Returns
        -------
        LocalExplanation with feature_contributions sorted by |SHAP| descending
        """
        self._ensure_built()
        X_scaled = self._prepare(sample)

        # shap_values shape: (n_classes, n_samples, n_features) in SHAP 0.40+
        shap_vals = self._explainer.shap_values(X_scaled)

        class_idx = self.class_names.index(predicted_class)

        # Handle both list-of-arrays and 3D array formats
        if isinstance(shap_vals, list):
            vals_for_class = shap_vals[class_idx][0]   # shape: (n_features,)
        else:
            vals_for_class = shap_vals[0, :, class_idx] if shap_vals.ndim == 3 else shap_vals[class_idx][0]

        contributions = [
            FeatureContribution(
                feature=feat,
                label=FEATURE_LABELS[feat],
                shap_value=float(vals_for_class[i]),
                direction="elevated" if vals_for_class[i] > 0 else "reduced",
                magnitude=float(abs(vals_for_class[i])),
            )
            for i, feat in enumerate(FEATURE_COLS)
        ]

        top_contributors = sorted(contributions, key=lambda c: c.magnitude, reverse=True)[:top_n]

        return LocalExplanation(
            target=self.target,
            predicted_class=predicted_class,
            feature_contributions=top_contributors,
            shap_values_raw=vals_for_class,
        )

    def explain_global(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """
        Compute mean absolute SHAP values across a dataset.

        Parameters
        ----------
        dataset : DataFrame containing FEATURE_COLS

        Returns
        -------
        DataFrame with columns [feature, mean_abs_shap, rank] sorted by importance
        """
        self._ensure_built()
        X_scaled = self.scaler.transform(dataset[FEATURE_COLS].fillna(0.0))
        shap_vals = self._explainer.shap_values(X_scaled)

        # Average across all classes: mean |SHAP| per feature
        if isinstance(shap_vals, list):
            mean_abs = np.mean([np.abs(sv).mean(axis=0) for sv in shap_vals], axis=0)
        else:
            mean_abs = np.abs(shap_vals).mean(axis=(0, 2)) if shap_vals.ndim == 3 else np.abs(shap_vals).mean(axis=0)

        df = pd.DataFrame({"feature": FEATURE_COLS, "mean_abs_shap": mean_abs})
        df = df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
        df["rank"] = df.index + 1
        return df

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _prepare(self, data: Union[dict, pd.DataFrame]) -> np.ndarray:
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        else:
            df = data.copy().reset_index(drop=True)
        return self.scaler.transform(df[FEATURE_COLS].fillna(0.0))

    def _ensure_built(self) -> None:
        if self._explainer is None:
            raise RuntimeError("Call build() before using the explainer.")
