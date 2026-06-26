"""
feature_extractor.py
Computes per-window physiological features from raw signals.

Window strategy: non-overlapping segments (default 60 s) at CHEST_FS = 700 Hz.

EDA features   : mean, std, peak_count, peak_amplitude
HRV features   : heart_rate_bpm, rmssd, sdnn
Resp features  : resp_rate_bpm, resp_variability
"""

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from typing import Optional

CHEST_FS = 700          # Hz - RespiBAN chest sensor
DEFAULT_WINDOW_S = 60   # seconds per feature window


class FeatureExtractor:
    """
    Segments raw signal arrays into fixed-length windows and extracts
    domain-specific features for EDA, ECG (HRV), and Respiration.
    """

    def __init__(self, fs: int = CHEST_FS, window_s: int = DEFAULT_WINDOW_S):
        self.fs = fs
        self.window_size = fs * window_s  # samples per window

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Slice df into windows and compute all features.

        Parameters
        ----------
        df : DataFrame with columns [ECG, EDA, Resp, label]

        Returns
        -------
        DataFrame with one row per window and all feature columns.
        """
        records = []
        for start in range(0, len(df) - self.window_size + 1, self.window_size):
            window = df.iloc[start : start + self.window_size]

            # Majority-vote label for the window
            dominant_label = window["label"].mode()[0]

            record = {"label": dominant_label}
            record.update(self._eda_features(window["EDA"].to_numpy()))
            record.update(self._hrv_features(window["ECG"].to_numpy()))
            record.update(self._resp_features(window["Resp"].to_numpy()))
            records.append(record)

        return pd.DataFrame(records)

    # ------------------------------------------------------------------
    # EDA features
    # ------------------------------------------------------------------

    def _eda_features(self, eda: np.ndarray) -> dict:
        peaks, props = find_peaks(eda, prominence=0.01)
        amplitudes = props["prominences"] if len(peaks) > 0 else np.array([0.0])

        return {
            "eda_mean":           float(np.mean(eda)),
            "eda_std":            float(np.std(eda)),
            "eda_peak_count":     int(len(peaks)),
            "eda_peak_amplitude": float(np.mean(amplitudes)),
        }

    # ------------------------------------------------------------------
    # HRV features (from ECG R-peaks)
    # ------------------------------------------------------------------

    def _hrv_features(self, ecg: np.ndarray) -> dict:
        r_peaks = self._detect_r_peaks(ecg)

        if len(r_peaks) < 2:
            return {"heart_rate_bpm": np.nan, "rmssd": np.nan, "sdnn": np.nan}

        # RR intervals in seconds
        rr = np.diff(r_peaks) / self.fs

        hr_bpm   = 60.0 / np.mean(rr)
        rmssd    = float(np.sqrt(np.mean(np.diff(rr) ** 2)))
        sdnn     = float(np.std(rr, ddof=1))

        return {
            "heart_rate_bpm": float(np.clip(hr_bpm, 30, 220)),
            "rmssd":          rmssd,
            "sdnn":           sdnn,
        }

    def _detect_r_peaks(self, ecg: np.ndarray) -> np.ndarray:
        """
        Lightweight R-peak detector using adaptive threshold on squared signal.
        Minimum distance enforced to avoid double-counting (HR cap ~220 bpm).
        """
        squared = ecg ** 2
        threshold = np.mean(squared) + np.std(squared)
        min_distance = int(self.fs * 60 / 220)  # samples between peaks
        peaks, _ = find_peaks(squared, height=threshold, distance=min_distance)
        return peaks

    # ------------------------------------------------------------------
    # Respiration features
    # ------------------------------------------------------------------

    def _resp_features(self, resp: np.ndarray) -> dict:
        # Detect breath cycles as peaks in the respiration signal
        min_dist = int(self.fs * 1.5)  # min 1.5 s between breaths (~40 bpm max)
        peaks, _ = find_peaks(resp, distance=min_dist)

        window_s = len(resp) / self.fs
        resp_rate = (len(peaks) / window_s) * 60.0 if window_s > 0 else np.nan

        # Variability = std of inter-breath intervals (seconds)
        if len(peaks) >= 2:
            ibi = np.diff(peaks) / self.fs
            variability = float(np.std(ibi))
        else:
            variability = np.nan

        return {
            "resp_rate_bpm":    float(np.clip(resp_rate, 3, 60)) if not np.isnan(resp_rate) else np.nan,
            "resp_variability": variability,
        }
