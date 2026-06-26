"""
responsible_ai.py
Top-level facade for the CogniArousal Responsible AI layer.

Wires together all Part 3 engines into a single, clean API:
    PredictionEngine      - model inference
    ExplainabilityEngine  - SHAP local/global explanations
    ConfidenceEngine      - probability → tier classification
    TransparencyEngine    - narrative + structured record assembly
    PrivacyEngine         - anonymous sessions + PII sanitisation

Usage
-----
    rai = ResponsibleAI(models_dir="models").load(background_df=feature_df)

    with rai.privacy.session() as session_id:
        result = rai.explain_prediction(
            session_id=session_id,
            target="cognitive_load",
            data={"eda_mean": 0.42, ...},
        )

    print(result.narrative)
    print(result.to_dict())

    global_df = rai.global_importance("arousal", dataset=feature_df)
"""

import logging
import pickle
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

from src.confidence_engine import ConfidenceEngine
from src.explainability_engine import ExplainabilityEngine
from src.prediction_engine import PredictionEngine
from src.privacy_engine import PrivacyEngine
from src.transparency_engine import ResponsibleAIPrediction, TransparencyEngine

logger = logging.getLogger(__name__)

FEATURE_COLS = [
    "eda_mean", "eda_std", "eda_peak_count", "eda_peak_amplitude",
    "heart_rate_bpm", "rmssd", "sdnn",
    "resp_rate_bpm", "resp_variability",
]

_TARGETS = ["arousal", "cognitive_load"]


class ResponsibleAI:
    """
    Unified Responsible AI inference interface.

    Parameters
    ----------
    models_dir : path containing arousal/ and cognitive_load/ artifact dirs
    """

    def __init__(self, models_dir: Union[str, Path] = "models"):
        self.models_dir = Path(models_dir)

        self.prediction   = PredictionEngine(models_dir)
        self.transparency = TransparencyEngine()
        self.privacy      = PrivacyEngine()

        # Per-target engines, initialised in load()
        self._explainers:  dict[str, ExplainabilityEngine] = {}
        self._confidence:  dict[str, ConfidenceEngine]     = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self, background_df: pd.DataFrame | None = None) -> "ResponsibleAI":
        """
        Load model artifacts and initialise all engines.

        Parameters
        ----------
        background_df : optional feature DataFrame used as SHAP background
                        distribution (recommended for more stable explanations).
                        If None, TreeExplainer uses tree_path_dependent mode.
        """
        self.prediction.load()

        for target in _TARGETS:
            art = self.prediction._artifacts[target]

            explainer = ExplainabilityEngine(
                model=art["model"],
                scaler=art["scaler"],
                class_names=list(art["label_encoder"].classes_),
                target=target,
            ).build(background=background_df)

            self._explainers[target] = explainer
            self._confidence[target] = ConfidenceEngine(
                class_names=list(art["label_encoder"].classes_)
            )

        logger.info("ResponsibleAI loaded for targets: %s", _TARGETS)
        return self

    # ------------------------------------------------------------------
    # Primary inference API
    # ------------------------------------------------------------------

    def explain_prediction(
        self,
        session_id: str,
        target: str,
        data: Union[dict, pd.DataFrame],
        top_n: int = 3,
        sanitise: bool = True,
    ) -> ResponsibleAIPrediction:
        """
        Run a fully transparent, responsible inference for one sample.

        Parameters
        ----------
        session_id : anonymous session ID (from privacy.session())
        target     : 'arousal' or 'cognitive_load'
        data       : dict or single-row DataFrame of physiological features
        top_n      : number of top SHAP contributors in the explanation
        sanitise   : if True, strip PII keys before processing

        Returns
        -------
        ResponsibleAIPrediction - predicted class, confidence, narrative, top features
        """
        self._check_loaded(target)

        # Privacy: strip any PII before anything touches the data
        if sanitise:
            data = self.privacy.sanitise(data, strict=False)

        # 1. Prediction
        pred_df   = self._predict(target, data)
        pred_cls  = pred_df["predicted_class"].iloc[0]
        proba_row = self._extract_proba_row(pred_df, target)

        # 2. Confidence
        confidence = self._confidence[target].evaluate(proba_row)

        # 3. Explanation
        explanation = self._explainers[target].explain_local(
            sample=data,
            predicted_class=pred_cls,
            top_n=top_n,
        )

        # 4. Transparency record
        return self.transparency.build_prediction(
            session_id=session_id,
            target=target,
            explanation=explanation,
            confidence=confidence,
        )

    def global_importance(
        self,
        target: str,
        dataset: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Compute mean |SHAP| importances across a dataset.

        Returns
        -------
        DataFrame with [feature, mean_abs_shap, rank]
        """
        self._check_loaded(target)
        return self._explainers[target].explain_global(dataset)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _predict(self, target: str, data: Union[dict, pd.DataFrame]) -> pd.DataFrame:
        if target == "arousal":
            return self.prediction.predict_emotional_arousal(data)
        return self.prediction.predict_cognitive_load(data)

    def _extract_proba_row(self, pred_df: pd.DataFrame, target: str) -> np.ndarray:
        """Extract probability array from prediction DataFrame in class order."""
        classes = self.prediction._artifacts[target]["label_encoder"].classes_
        return np.array([pred_df[f"prob_{c}"].iloc[0] for c in classes])

    def _check_loaded(self, target: str) -> None:
        if target not in self._explainers:
            raise RuntimeError(
                f"Target '{target}' not loaded. Call load() first."
            )
