"""
confidence_engine.py
Classifies model prediction confidence from output probabilities.

Tiers (based on max class probability):
    High    > 85%   - reliable prediction, act on it
    Medium  60–85%  - moderate certainty, use with caution
    Low     < 60%   - uncertain prediction, flag for review
"""

from dataclasses import dataclass
from typing import Union

import numpy as np
import pandas as pd

# Tier thresholds (inclusive lower bound)
_HIGH_THRESHOLD   = 0.85
_MEDIUM_THRESHOLD = 0.60


@dataclass
class ConfidenceResult:
    confidence_score: float   # max class probability [0, 1]
    confidence_pct:   float   # confidence_score × 100
    tier:             str     # "High" | "Medium" | "Low"
    flag_review:      bool    # True when tier is Low
    entropy:          float   # prediction entropy - higher = more uncertain
    class_probs:      dict[str, float]  # {class_name: probability}


class ConfidenceEngine:
    """
    Converts raw model class probabilities into a structured ConfidenceResult.

    Parameters
    ----------
    class_names : list of class names in the same order as model.predict_proba() output
    """

    def __init__(self, class_names: list[str]):
        self.class_names = class_names

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        probabilities: Union[np.ndarray, list[float]],
    ) -> ConfidenceResult:
        """
        Evaluate confidence from a 1-D probability vector.

        Parameters
        ----------
        probabilities : array-like of shape (n_classes,)

        Returns
        -------
        ConfidenceResult
        """
        probs = np.asarray(probabilities, dtype=float)
        if probs.ndim != 1 or len(probs) != len(self.class_names):
            raise ValueError(
                f"Expected 1-D array of length {len(self.class_names)}, got shape {probs.shape}."
            )

        confidence = float(probs.max())
        tier        = self._tier(confidence)

        return ConfidenceResult(
            confidence_score = confidence,
            confidence_pct   = round(confidence * 100, 2),
            tier             = tier,
            flag_review      = tier == "Low",
            entropy          = float(self._entropy(probs)),
            class_probs      = {c: round(float(p), 4) for c, p in zip(self.class_names, probs)},
        )

    def evaluate_batch(self, prob_matrix: np.ndarray) -> list[ConfidenceResult]:
        """
        Evaluate confidence for multiple samples at once.

        Parameters
        ----------
        prob_matrix : array of shape (n_samples, n_classes)

        Returns
        -------
        list of ConfidenceResult, one per row
        """
        return [self.evaluate(row) for row in prob_matrix]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _tier(confidence: float) -> str:
        if confidence > _HIGH_THRESHOLD:
            return "High"
        if confidence >= _MEDIUM_THRESHOLD:
            return "Medium"
        return "Low"

    @staticmethod
    def _entropy(probs: np.ndarray) -> float:
        """Shannon entropy - measures prediction uncertainty (higher = less certain)."""
        clipped = np.clip(probs, 1e-12, 1.0)
        return float(-np.sum(clipped * np.log2(clipped)))
