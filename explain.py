"""
explain.py
Part 3 -- Responsible AI Engine demo and validation script.

Exercises:
    - Anonymous session lifecycle (PrivacyEngine)
    - Transparent prediction with narrative (TransparencyEngine)
    - Local SHAP explanation for arousal + cognitive load (ExplainabilityEngine)
    - Confidence tier classification (ConfidenceEngine)
    - Global SHAP feature importances across full dataset

Usage:
    python explain.py
    python explain.py --feature_csv output/feature_dataset.csv --models_dir models/
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from src.responsible_ai import ResponsibleAI

# Force UTF-8 output on Windows to avoid cp1252 encoding errors
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

TARGETS = ["arousal", "cognitive_load"]
FEATURE_COLS = [
    "eda_mean", "eda_std", "eda_peak_count", "eda_peak_amplitude",
    "heart_rate_bpm", "rmssd", "sdnn", "resp_rate_bpm", "resp_variability",
]


def print_prediction(result) -> None:
    sep = "-" * 62
    print(f"\n{sep}")
    print(f"  Target          : {result.target.upper()}")
    print(f"  Session         : {result.session_id}")
    print(f"  Predicted Class : {result.predicted_class}")
    print(f"  Confidence      : {result.confidence.confidence_pct:.1f}%  [{result.confidence.tier}]")
    print(f"  Entropy         : {result.confidence.entropy:.4f}")
    print(f"  Flag Review     : {result.flag_review}")
    print(f"\n  Narrative:")
    print(f"    {result.narrative}")
    print(f"\n  Top Contributing Features:")
    for fc in result.top_features:
        direction_tag = "(+)" if fc.direction == "elevated" else "(-)"
        print(f"    {direction_tag} {fc.label:<28}  SHAP={fc.shap_value:+.4f}")
    print(f"\n  Class Probabilities:")
    for cls, prob in result.confidence.class_probs.items():
        bar = "#" * int(prob * 30)
        print(f"    {cls:<10}  {prob*100:5.1f}%  {bar}")
    print(sep)


def print_global_importance(target: str, df: pd.DataFrame) -> None:
    sep = "-" * 62
    print(f"\n{sep}")
    print(f"  Global SHAP Feature Importance -- {target.upper()}")
    print(sep)
    for _, row in df.iterrows():
        bar = "#" * int(row["mean_abs_shap"] * 200)
        print(f"  {int(row['rank']):>2}. {row['feature']:<25}  {row['mean_abs_shap']:.4f}  {bar}")
    print(sep)


def run(feature_csv: Path, models_dir: Path) -> None:
    df = pd.read_csv(feature_csv)
    logger.info("Loaded feature dataset: %d rows", len(df))

    rai = ResponsibleAI(models_dir=models_dir).load(background_df=df)

    # One sample per WESAD label class
    samples = {
        label: df[df["label"] == label].iloc[0]
        for label in df["label"].unique()
    }

    print("\n" + "=" * 62)
    print("  CogniArousal -- Part 3: Responsible AI Predictions")
    print("=" * 62)

    for label, row in samples.items():
        sample_dict = row[FEATURE_COLS].to_dict()
        print(f"\n  [WESAD Label: {label}]")

        for target in TARGETS:
            with rai.privacy.session() as session_id:
                result = rai.explain_prediction(
                    session_id=session_id,
                    target=target,
                    data=sample_dict,
                    top_n=3,
                    sanitise=True,
                )
            print_prediction(result)

    # --- Global SHAP importances ---
    print("\n" + "=" * 62)
    print("  Global Feature Importance (SHAP)")
    print("=" * 62)

    for target in TARGETS:
        global_df = rai.global_importance(target=target, dataset=df)
        print_global_importance(target, global_df)

        out_path = models_dir / target / "shap_global_importance.csv"
        global_df.to_csv(out_path, index=False)
        logger.info("Saved global SHAP importances -> %s", out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="CogniArousal -- Part 3: Responsible AI Engine")
    parser.add_argument("--feature_csv", type=Path, default=Path("output/feature_dataset.csv"))
    parser.add_argument("--models_dir",  type=Path, default=Path("models"))
    args = parser.parse_args()
    run(args.feature_csv, args.models_dir)


if __name__ == "__main__":
    main()
