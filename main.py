"""
main.py
Pipeline orchestrator for CogniArousal Part 1.

Usage:
    python main.py --data_dir data/ --output_dir output/

Expected data layout:
    data/
        S2/S2.pkl
        S3/S3.pkl
        ...  (WESAD subjects S2–S17, S1 excluded per dataset notes)

Outputs:
    output/physiological_data.csv   - raw aligned signals + labels (all subjects)
    output/feature_dataset.csv      - cleaned, normalized feature set for ML
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

from src.dataset_processor import DatasetProcessor
from src.feature_extractor import FeatureExtractor
from src.data_cleaner import DataCleaner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def process_subject(pkl_path: Path, extractor: FeatureExtractor) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Full per-subject pipeline:
        load → extract signals → build signal df → extract features

    Returns
    -------
    signal_df  : sample-level raw signals with labels
    feature_df : window-level feature vectors with labels
    """
    subject_id = pkl_path.parent.name

    processor = DatasetProcessor(pkl_path)
    processor.load().extract_signals()
    signal_df = processor.get_labeled_dataframe()
    signal_df.insert(0, "subject_id", subject_id)

    feature_df = extractor.extract(signal_df)
    feature_df.insert(0, "subject_id", subject_id)

    return signal_df, feature_df


def discover_subjects(data_dir: Path) -> list[Path]:
    """Find all subject pickle files under data_dir (pattern: S*/S*.pkl)."""
    paths = sorted(data_dir.glob("S*/S*.pkl"))
    if not paths:
        raise FileNotFoundError(
            f"No WESAD subject files found in '{data_dir}'.\n"
            "Expected layout: data/S2/S2.pkl, data/S3/S3.pkl, ..."
        )
    logger.info("Found %d subject file(s): %s", len(paths), [p.parent.name for p in paths])
    return paths


def run(data_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    subject_paths = discover_subjects(data_dir)
    extractor = FeatureExtractor()
    cleaner = DataCleaner()

    all_signals: list[pd.DataFrame] = []
    all_features: list[pd.DataFrame] = []

    for pkl_path in subject_paths:
        try:
            sig_df, feat_df = process_subject(pkl_path, extractor)
            all_signals.append(sig_df)
            all_features.append(feat_df)
            logger.info("%s → %d windows extracted.", pkl_path.parent.name, len(feat_df))
        except Exception as exc:  # noqa: BLE001
            logger.error("Skipping %s - %s", pkl_path.parent.name, exc)

    if not all_features:
        logger.error("No subjects processed successfully. Exiting.")
        return

    # --- Combine & export raw physiological data ---
    physiological_df = pd.concat(all_signals, ignore_index=True)
    phys_path = output_dir / "physiological_data.csv"
    physiological_df.to_csv(phys_path, index=False)
    logger.info("Saved physiological data → %s  (%d rows)", phys_path, len(physiological_df))

    # --- Combine, clean, normalize & export feature dataset ---
    raw_features_df = pd.concat(all_features, ignore_index=True)
    feature_df = cleaner.fit_transform(raw_features_df)
    feat_path = output_dir / "feature_dataset.csv"
    feature_df.to_csv(feat_path, index=False)
    logger.info("Saved feature dataset    → %s  (%d rows, %d features)",
                feat_path, len(feature_df), len(feature_df.columns) - 2)  # subtract label + subject_id

    # --- Summary ---
    logger.info("\n=== Label Distribution (feature dataset) ===\n%s",
                feature_df["label"].value_counts().to_string())


def main() -> None:
    parser = argparse.ArgumentParser(description="CogniArousal - Part 1: Physiological Data Processing")
    parser.add_argument("--data_dir",   type=Path, default=Path("data"),   help="Directory containing WESAD subject folders")
    parser.add_argument("--output_dir", type=Path, default=Path("output"), help="Directory for exported CSV files")
    args = parser.parse_args()

    run(args.data_dir, args.output_dir)


if __name__ == "__main__":
    main()
