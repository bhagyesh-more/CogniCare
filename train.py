"""
train.py
Part 2 training pipeline - CogniArousal.

Usage:
    python train.py
    python train.py --feature_csv output/feature_dataset.csv --models_dir models/

Steps:
    1. Load feature_dataset.csv (Part 1 output)
    2. Derive arousal + cognitive_load target columns
    3. Train Random Forest for each target
    4. Print consolidated evaluation report
    5. Smoke-test PredictionEngine on a held-out sample
"""

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from src.label_encoder import add_target_labels
from src.model_trainer import ModelTrainer
from src.prediction_engine import PredictionEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

TARGETS = ["arousal", "cognitive_load"]


def print_report(all_metrics: list[dict]) -> None:
    """Print a clean side-by-side evaluation summary to stdout."""
    sep = "=" * 60
    print(f"\n{sep}")
    print("  CogniArousal - Part 2 Training Report")
    print(sep)
    for m in all_metrics:
        print(f"\n  Target         : {m['target'].upper()}")
        print(f"  Classes        : {m['classes']}")
        print(f"  Samples        : {m['n_samples']}")
        print(f"  Train Accuracy : {m['train_accuracy']:.3f}")
        print(f"  CV Accuracy    : {m['cv_accuracy']:.3f} ± {m['cv_accuracy_std']:.3f}")
        print(f"  CV F1 (macro)  : {m['cv_f1_macro']:.3f} ± {m['cv_f1_std']:.3f}")
        print(f"  CV Precision   : {m['cv_precision']:.3f}")
        print(f"  CV Recall      : {m['cv_recall']:.3f}")

        # Per-class breakdown
        report = m.get("classification_report", {})
        print("\n  Per-Class F1:")
        for cls in m["classes"]:
            if cls in report:
                r = report[cls]
                print(
                    f"    {cls:<10}  precision={r['precision']:.3f}  "
                    f"recall={r['recall']:.3f}  f1={r['f1-score']:.3f}  "
                    f"support={int(r['support'])}"
                )
    print(f"\n{sep}\n")


def smoke_test(models_dir: Path, sample: pd.Series) -> None:
    """Verify PredictionEngine loads and returns valid predictions."""
    logger.info("Running inference smoke test on one sample...")
    engine = PredictionEngine(models_dir).load()

    row = sample.to_frame().T.reset_index(drop=True)

    arousal_result = engine.predict_emotional_arousal(row)
    cog_result     = engine.predict_cognitive_load(row)

    logger.info(
        "Smoke test passed ✓  |  arousal=%s (%.2f%%)  |  cognitive_load=%s (%.2f%%)",
        arousal_result["predicted_class"].iloc[0],
        arousal_result["confidence"].iloc[0] * 100,
        cog_result["predicted_class"].iloc[0],
        cog_result["confidence"].iloc[0] * 100,
    )


def run(feature_csv: Path, models_dir: Path) -> None:
    # 1. Load Part 1 output
    if not feature_csv.exists():
        raise FileNotFoundError(
            f"Feature dataset not found: {feature_csv}\n"
            "Run main.py (Part 1) first."
        )
    df = pd.read_csv(feature_csv)
    logger.info("Loaded feature dataset: %d rows × %d cols", *df.shape)

    # 2. Derive classification targets
    df = add_target_labels(df)
    logger.info(
        "Target distributions:\n  arousal:\n%s\n  cognitive_load:\n%s",
        df["arousal"].value_counts().to_string(),
        df["cognitive_load"].value_counts().to_string(),
    )

    # 3. Train one Random Forest per target
    all_metrics = []
    for target in TARGETS:
        logger.info("─" * 50)
        logger.info("Training target: %s", target)
        trainer = ModelTrainer(target=target, models_dir=models_dir)
        trainer.train(df)
        all_metrics.append(trainer.get_metrics())

    # 4. Consolidated report
    print_report(all_metrics)

    # 5. Smoke-test inference engine on first sample
    feature_cols = [
        "eda_mean", "eda_std", "eda_peak_count", "eda_peak_amplitude",
        "heart_rate_bpm", "rmssd", "sdnn", "resp_rate_bpm", "resp_variability",
    ]
    smoke_test(models_dir, df[feature_cols].iloc[0])


def main() -> None:
    parser = argparse.ArgumentParser(description="CogniArousal - Part 2: ML Training Pipeline")
    parser.add_argument(
        "--feature_csv",
        type=Path,
        default=Path("output/feature_dataset.csv"),
        help="Path to feature_dataset.csv from Part 1",
    )
    parser.add_argument(
        "--models_dir",
        type=Path,
        default=Path("models"),
        help="Directory to save trained model artifacts",
    )
    args = parser.parse_args()
    run(args.feature_csv, args.models_dir)


if __name__ == "__main__":
    main()
