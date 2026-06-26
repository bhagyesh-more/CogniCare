"""
label_encoder.py
Derives two classification targets from WESAD affect labels.

Psychophysiology basis
----------------------
Emotional Arousal  (activation dimension of Russell's circumplex model):
    stress    -> High    (peak sympathetic activation, fight-or-flight)
    amusement -> High    (laughter/engagement drives elevated HR and EDA)
    baseline  -> Low     (resting parasympathetic state)

Cognitive Load  (mental workload, Sweller 1988 / NASA-TLX proxy):
    stress    -> High    (dual cognitive + physiological demand)
    amusement -> Low     (passive film-viewing = minimal working memory load)
    baseline  -> Low     (no task demand)

Key difference: amusement maps to HIGH arousal but LOW cognitive load.
This creates genuinely distinct training targets so the two classifiers
learn different decision boundaries and produce different predictions.
"""

import pandas as pd

# Arousal: amusement = High (active positive affect, elevated HR/EDA)
AROUSAL_MAP: dict[str, str] = {
    "baseline":  "Low",
    "amusement": "High",
    "stress":    "High",
}

# Cognitive Load: amusement = Low (passive entertainment, minimal working memory)
COGNITIVE_LOAD_MAP: dict[str, str] = {
    "baseline":  "Low",
    "amusement": "Low",
    "stress":    "High",
}


def add_target_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Append `arousal` and `cognitive_load` columns derived from the
    WESAD `label` column.  Original `label` column is retained.
    """
    if "label" not in df.columns:
        raise ValueError("DataFrame must contain a 'label' column.")

    df = df.copy()
    df["arousal"]        = df["label"].map(AROUSAL_MAP)
    df["cognitive_load"] = df["label"].map(COGNITIVE_LOAD_MAP)

    unmapped = df["arousal"].isna().sum()
    if unmapped:
        raise ValueError(f"{unmapped} rows could not be mapped - unexpected label values.")

    return df
