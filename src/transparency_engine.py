"""
transparency_engine.py
Assembles a full transparent prediction record for every inference call.

Combines:
    - Predicted class (from PredictionEngine)
    - Confidence tier (from ConfidenceEngine)
    - Top contributing features (from ExplainabilityEngine)
    - Human-readable narrative

Example output narrative:
    "High cognitive load predicted with high confidence (91.2%), primarily
     driven by elevated heart rate and elevated EDA variability."
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from src.confidence_engine import ConfidenceResult
from src.explainability_engine import FeatureContribution, LocalExplanation

# Target display names for narrative
_TARGET_DISPLAY = {
    "arousal":        "emotional arousal",
    "cognitive_load": "cognitive load",
}


@dataclass
class ResponsibleAIPrediction:
    """Complete transparent prediction record."""
    session_id:       str
    target:           str
    predicted_class:  str
    confidence:       ConfidenceResult
    top_features:     list[FeatureContribution]
    narrative:        str
    flag_review:      bool

    def to_dict(self) -> dict:
        return {
            "session_id":      self.session_id,
            "target":          self.target,
            "predicted_class": self.predicted_class,
            "confidence_pct":  self.confidence.confidence_pct,
            "confidence_tier": self.confidence.tier,
            "entropy":         self.confidence.entropy,
            "flag_review":     self.flag_review,
            "narrative":       self.narrative,
            "top_features": [
                {
                    "feature":    fc.feature,
                    "label":      fc.label,
                    "direction":  fc.direction,
                    "shap_value": round(fc.shap_value, 5),
                }
                for fc in self.top_features
            ],
            "class_probabilities": self.confidence.class_probs,
        }


class TransparencyEngine:
    """
    Builds a ResponsibleAIPrediction from the outputs of the other engines.

    This is intentionally stateless - all context is passed per call so it
    can be used in multi-threaded / multi-session environments.
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_prediction(
        self,
        session_id:      str,
        target:          str,
        explanation:     LocalExplanation,
        confidence:      ConfidenceResult,
    ) -> ResponsibleAIPrediction:
        """
        Assemble a full transparent prediction record.

        Parameters
        ----------
        session_id  : anonymous session identifier
        target      : 'arousal' or 'cognitive_load'
        explanation : LocalExplanation from ExplainabilityEngine
        confidence  : ConfidenceResult from ConfidenceEngine

        Returns
        -------
        ResponsibleAIPrediction
        """
        narrative = self._generate_narrative(
            target=target,
            predicted_class=explanation.predicted_class,
            confidence=confidence,
            top_features=explanation.feature_contributions,
        )

        return ResponsibleAIPrediction(
            session_id      = session_id,
            target          = target,
            predicted_class = explanation.predicted_class,
            confidence      = confidence,
            top_features    = explanation.feature_contributions,
            narrative       = narrative,
            flag_review     = confidence.flag_review,
        )

    # ------------------------------------------------------------------
    # Private - narrative generation
    # ------------------------------------------------------------------

    @staticmethod
    def _generate_narrative(
        target:          str,
        predicted_class: str,
        confidence:      ConfidenceResult,
        top_features:    list[FeatureContribution],
    ) -> str:
        target_label = _TARGET_DISPLAY.get(target, target.replace("_", " "))
        confidence_phrase = (
            f"with {confidence.tier.lower()} confidence ({confidence.confidence_pct:.1f}%)"
        )

        if not top_features:
            return f"{predicted_class} {target_label} predicted {confidence_phrase}."

        # Build feature phrase: "elevated heart rate, reduced HRV (RMSSD), and elevated EDA variability"
        parts = []
        for fc in top_features:
            parts.append(f"{fc.direction} {fc.label}")

        if len(parts) == 1:
            feature_phrase = parts[0]
        elif len(parts) == 2:
            feature_phrase = f"{parts[0]} and {parts[1]}"
        else:
            feature_phrase = ", ".join(parts[:-1]) + f", and {parts[-1]}"

        narrative = (
            f"{predicted_class} {target_label} predicted {confidence_phrase}, "
            f"primarily driven by {feature_phrase}."
        )

        # Append review flag notice
        if confidence.flag_review:
            narrative += " [LOW CONFIDENCE] Manual review recommended."

        return narrative
