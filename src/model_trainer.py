"""
model_trainer.py
Trains a Random Forest classifier for a single target, evaluates it,
persists model artifacts, and produces a feature importance report.

Per-target artifacts saved under models/<target>/:
    model.pkl               - fitted RandomForestClassifier
    scaler.pkl              - fitted StandardScaler (re-fit on train split)
    feature_metadata.json   - feature names, classes, split sizes, metrics
    feature_importance.csv  - ranked feature importances
    confusion_matrix.csv    - raw confusion matrix
"""

import json
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import LabelEncoder, StandardScaler

logger = logging.getLogger(__name__)

FEATURE_COLS = [
    "eda_mean", "eda_std", "eda_peak_count", "eda_peak_amplitude",
    "heart_rate_bpm", "rmssd", "sdnn",
    "resp_rate_bpm", "resp_variability",
]

RF_PARAMS = {
    "n_estimators":      300,
    "max_depth":         None,
    "min_samples_split": 4,
    "min_samples_leaf":  2,
    "class_weight":      "balanced",   # handles label imbalance
    "random_state":      42,
    "n_jobs":            -1,
}


class ModelTrainer:
    """
    Trains and evaluates a Random Forest classifier for one target column.

    Parameters
    ----------
    target   : column name in df - 'arousal' or 'cognitive_load'
    models_dir : root output directory; artifacts saved under models_dir/target/
    """

    def __init__(self, target: str, models_dir: Path = Path("models")):
        self.target = target
        self.artifact_dir = models_dir / target
        self.artifact_dir.mkdir(parents=True, exist_ok=True)

        self.le      = LabelEncoder()
        self.scaler  = StandardScaler()
        self.model   = RandomForestClassifier(**RF_PARAMS)
        self.metrics: dict = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(self, df: pd.DataFrame) -> "ModelTrainer":
        """
        Full training pipeline:
            prepare → cross-validate → final fit on full data → evaluate → persist
        """
        X, y = self._prepare(df)

        # Stratified 5-fold CV for robust estimates on small dataset
        cv_results = self._cross_validate(X, y)
        logger.info(
            "[%s] CV macro-F1: %.3f ± %.3f",
            self.target,
            cv_results["test_f1"].mean(),
            cv_results["test_f1"].std(),
        )

        # Final fit on full dataset - model will be used for inference
        self.model.fit(X, y)
        y_pred = self.model.predict(X)

        self.metrics = self._compute_metrics(y, y_pred, cv_results)
        self._log_metrics()

        self._save_artifacts(X, y, y_pred)
        return self

    def get_metrics(self) -> dict:
        return self.metrics

    # ------------------------------------------------------------------
    # Private - data preparation
    # ------------------------------------------------------------------

    def _prepare(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Extract feature matrix and encode target labels."""
        missing = [c for c in FEATURE_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"Missing feature columns: {missing}")

        X_raw = df[FEATURE_COLS].copy()

        # Impute any residual NaN with column medians (safety net)
        X_raw = X_raw.fillna(X_raw.median())

        X = self.scaler.fit_transform(X_raw)
        y = self.le.fit_transform(df[self.target])
        return X, y

    # ------------------------------------------------------------------
    # Private - evaluation
    # ------------------------------------------------------------------

    def _cross_validate(self, X: np.ndarray, y: np.ndarray) -> dict:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        return cross_validate(
            self.model, X, y, cv=cv,
            scoring={
                "accuracy": "accuracy",
                "f1":       "f1_macro",
                "precision":"precision_macro",
                "recall":   "recall_macro",
            },
            return_train_score=False,
        )

    def _compute_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray, cv: dict
    ) -> dict:
        return {
            "target":          self.target,
            "classes":         list(self.le.classes_),
            "n_samples":       int(len(y_true)),
            "train_accuracy":  float(accuracy_score(y_true, y_pred)),
            "cv_accuracy":     float(cv["test_accuracy"].mean()),
            "cv_accuracy_std": float(cv["test_accuracy"].std()),
            "cv_f1_macro":     float(cv["test_f1"].mean()),
            "cv_f1_std":       float(cv["test_f1"].std()),
            "cv_precision":    float(cv["test_precision"].mean()),
            "cv_recall":       float(cv["test_recall"].mean()),
            "classification_report": classification_report(
                y_true, y_pred,
                target_names=self.le.classes_,
                output_dict=True,
            ),
        }

    def _log_metrics(self) -> None:
        m = self.metrics
        logger.info(
            "[%s] Train accuracy: %.3f | CV accuracy: %.3f ± %.3f | CV F1: %.3f ± %.3f",
            self.target,
            m["train_accuracy"],
            m["cv_accuracy"], m["cv_accuracy_std"],
            m["cv_f1_macro"], m["cv_f1_std"],
        )

    # ------------------------------------------------------------------
    # Private - persistence
    # ------------------------------------------------------------------

    def _save_artifacts(
        self, X: np.ndarray, y: np.ndarray, y_pred: np.ndarray
    ) -> None:
        """Persist model, scaler, metadata, importances, confusion matrix."""

        # model.pkl
        with open(self.artifact_dir / "model.pkl", "wb") as f:
            pickle.dump({"model": self.model, "label_encoder": self.le}, f)

        # scaler.pkl
        with open(self.artifact_dir / "scaler.pkl", "wb") as f:
            pickle.dump(self.scaler, f)

        # feature_metadata.json
        metadata = {
            "target":        self.target,
            "feature_cols":  FEATURE_COLS,
            "classes":       list(self.le.classes_),
            "n_samples":     int(len(y)),
            "rf_params":     RF_PARAMS,
            "metrics": {
                k: v for k, v in self.metrics.items()
                if k != "classification_report"   # keep JSON lean; full report in csv
            },
        }
        with open(self.artifact_dir / "feature_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        # feature_importance.csv  (ranked)
        importance_df = pd.DataFrame(
            {
                "feature":   FEATURE_COLS,
                "importance": self.model.feature_importances_,
                "std":        np.std(
                    [t.feature_importances_ for t in self.model.estimators_], axis=0
                ),
            }
        ).sort_values("importance", ascending=False).reset_index(drop=True)
        importance_df["rank"] = importance_df.index + 1
        importance_df.to_csv(self.artifact_dir / "feature_importance.csv", index=False)

        # confusion_matrix.csv
        cm = confusion_matrix(y, y_pred)
        cm_df = pd.DataFrame(cm, index=self.le.classes_, columns=self.le.classes_)
        cm_df.to_csv(self.artifact_dir / "confusion_matrix.csv")

        logger.info(
            "[%s] Artifacts saved → %s",
            self.target, self.artifact_dir.resolve(),
        )

        # Print feature importance report to console
        self._print_importance_report(importance_df)

    def _print_importance_report(self, df: pd.DataFrame) -> None:
        header = f"\n{'─'*50}\nFeature Importance Report - {self.target.upper()}\n{'─'*50}"
        rows = "\n".join(
            f"  {r['rank']:>2}. {r['feature']:<25}  {r['importance']:.4f}  ±{r['std']:.4f}"
            for _, r in df.iterrows()
        )
        logger.info("%s\n%s\n%s", header, rows, "─" * 50)
