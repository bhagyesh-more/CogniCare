"""
data_cleaner.py
Handles outlier removal, missing value imputation, and feature normalization.

Strategy:
    - Outliers: IQR-based capping (Winsorization) per numeric column
    - Missing values: median imputation per column
    - Normalization: StandardScaler (zero mean, unit variance) on feature columns
"""

import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# Columns excluded from numeric cleaning / normalization
NON_FEATURE_COLS = {"label", "subject_id"}


class DataCleaner:
    """
    Cleans a feature DataFrame in three stages:
        1. IQR-based outlier capping
        2. Median imputation for NaN values
        3. StandardScaler normalization
    """

    def __init__(self, iqr_factor: float = 1.5):
        self.iqr_factor = iqr_factor
        self.scaler = StandardScaler()
        self._feature_cols: list[str] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and normalize; fits the scaler on this data."""
        df = df.copy()
        self._feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]

        df = self._remove_outliers(df)
        df = self._impute_missing(df)
        df = self._normalize(df, fit=True)
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply previously fitted cleaning params to new data."""
        if not self._feature_cols:
            raise RuntimeError("Call fit_transform before transform.")
        df = df.copy()
        df = self._remove_outliers(df)
        df = self._impute_missing(df)
        df = self._normalize(df, fit=False)
        return df

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cap values outside [Q1 - k*IQR, Q3 + k*IQR] to the fence value."""
        for col in self._feature_cols:
            q1, q3 = df[col].quantile([0.25, 0.75])
            iqr = q3 - q1
            lower, upper = q1 - self.iqr_factor * iqr, q3 + self.iqr_factor * iqr
            n_clipped = ((df[col] < lower) | (df[col] > upper)).sum()
            if n_clipped:
                logger.debug("Outlier capping - %s: %d values clipped.", col, n_clipped)
            df[col] = df[col].clip(lower=lower, upper=upper)
        return df

    def _impute_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill NaN with column median."""
        missing = df[self._feature_cols].isna().sum()
        imputed_cols = missing[missing > 0]
        if not imputed_cols.empty:
            logger.warning("Imputing missing values:\n%s", imputed_cols.to_string())

        medians = df[self._feature_cols].median()
        df[self._feature_cols] = df[self._feature_cols].fillna(medians)
        return df

    def _normalize(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        """Apply StandardScaler to feature columns."""
        if fit:
            df[self._feature_cols] = self.scaler.fit_transform(df[self._feature_cols])
        else:
            df[self._feature_cols] = self.scaler.transform(df[self._feature_cols])
        return df
