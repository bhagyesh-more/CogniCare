"""
src/monitoring.py
Lightweight monitoring layer for CogniArousal inference pipeline.

Tracks:
    - Model load status per target
    - Per-inference timing (wall-clock ms)
    - Rolling inference count and average latency
    - Dataset availability status

All state is held in a simple dict - designed to be stored in
Streamlit session_state so it survives page rerenders.
"""

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator

MODELS_DIR = Path("models")
FEATURE_CSV = Path("output/feature_dataset.csv")


@dataclass
class InferenceMetrics:
    inference_count:   int   = 0
    total_latency_ms:  float = 0.0
    last_latency_ms:   float = 0.0
    min_latency_ms:    float = float("inf")
    max_latency_ms:    float = 0.0

    @property
    def avg_latency_ms(self) -> float:
        if self.inference_count == 0:
            return 0.0
        return self.total_latency_ms / self.inference_count

    def record(self, ms: float) -> None:
        self.inference_count  += 1
        self.total_latency_ms += ms
        self.last_latency_ms   = ms
        self.min_latency_ms    = min(self.min_latency_ms, ms)
        self.max_latency_ms    = max(self.max_latency_ms, ms)


class Monitor:
    """
    Runtime monitoring for model load status, dataset status, and inference timing.
    Instantiate once and store in st.session_state.
    """

    def __init__(self) -> None:
        self.metrics = InferenceMetrics()
        self._load_start: float = 0.0

    # ------------------------------------------------------------------
    # Status checks
    # ------------------------------------------------------------------

    def model_status(self) -> dict[str, str]:
        """
        Returns per-target artifact status.
        Status values: "OK" | "MISSING"
        """
        status = {}
        for target in ("arousal", "cognitive_load"):
            target_dir = MODELS_DIR / target
            all_present = all(
                (target_dir / f).exists()
                for f in ("model.pkl", "scaler.pkl", "feature_metadata.json")
            )
            status[target] = "OK" if all_present else "MISSING"
        return status

    def dataset_status(self) -> dict[str, str]:
        """Check feature dataset and physiological data availability."""
        return {
            "feature_dataset":     "OK" if FEATURE_CSV.exists() else "MISSING",
            "physiological_data":  "OK" if Path("output/physiological_data.csv").exists() else "MISSING",
        }

    def system_status(self) -> dict[str, bool]:
        """Consolidated boolean status for all critical components."""
        ms  = self.model_status()
        ds  = self.dataset_status()
        return {
            "models_ok":  all(v == "OK" for v in ms.values()),
            "dataset_ok": ds["feature_dataset"] == "OK",
            "all_ok":     all(v == "OK" for v in ms.values()) and ds["feature_dataset"] == "OK",
        }

    # ------------------------------------------------------------------
    # Timing
    # ------------------------------------------------------------------

    @contextmanager
    def time_inference(self) -> Generator[None, None, None]:
        """
        Context manager that measures inference wall-clock time and
        records it in self.metrics.

        Usage:
            with monitor.time_inference():
                result = rai.explain_prediction(...)
        """
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000
            self.metrics.record(elapsed_ms)

    def record_load_time(self, start_time: float) -> float:
        """Record model load duration in ms."""
        elapsed = (time.perf_counter() - start_time) * 1000
        return elapsed

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> dict:
        return {
            "inference_count":   self.metrics.inference_count,
            "avg_latency_ms":    round(self.metrics.avg_latency_ms, 1),
            "last_latency_ms":   round(self.metrics.last_latency_ms, 1),
            "min_latency_ms":    round(self.metrics.min_latency_ms, 1) if self.metrics.min_latency_ms != float("inf") else 0,
            "max_latency_ms":    round(self.metrics.max_latency_ms, 1),
            **self.model_status(),
            **self.dataset_status(),
        }
