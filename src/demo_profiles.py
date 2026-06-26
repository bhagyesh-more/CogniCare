"""
src/demo_profiles.py
Built-in physiological profiles for instant platform demonstration.

All values are z-scores (StandardScaler from WESAD training data).
Profiles are derived from empirical WESAD label centroids:
    - baseline  → Low arousal / Low cognitive load
    - stress    → High arousal / High cognitive load
    - amusement → High arousal / Low cognitive load (key distinction!)

Each profile carries human-readable clinical annotations for display.

Note: The models predict binary classes (High/Low), not ternary (Low/Medium/High).
The amusement condition is crucial: it has HIGH emotional arousal (laughter/engagement
drives elevated HR/EDA) but LOW cognitive load (passive film-viewing requires minimal
working memory). This creates genuinely distinct predictions from the stress condition.
"""

from dataclasses import dataclass, field

FEATURE_COLS = [
    "eda_mean", "eda_std", "eda_peak_count", "eda_peak_amplitude",
    "heart_rate_bpm", "rmssd", "sdnn", "resp_rate_bpm", "resp_variability",
]


@dataclass
class DemoProfile:
    key:         str
    name:        str
    description: str
    icon:        str
    accent:      str           # hex colour for UI
    features:    dict[str, float] = field(default_factory=dict)
    expected_arousal:    str = ""
    expected_cog_load:   str = ""
    clinical_notes:      list[str] = field(default_factory=list)


# ── Six canonical profiles ────────────────────────────────────────────────────

PROFILES: dict[str, DemoProfile] = {

    "low_cognitive_load": DemoProfile(
        key="low_cognitive_load",
        name="Low Cognitive Load",
        description="Resting baseline state - minimal mental demand",
        icon="◎",
        accent="#00FFB2",
        expected_arousal="Low",
        expected_cog_load="Low",
        features={
            "eda_mean":           -0.45,   # below-mean skin conductance
            "eda_std":            -0.60,   # stable EDA signal
            "eda_peak_count":     -0.52,   # few arousal spikes
            "eda_peak_amplitude": -0.48,   # low amplitude responses
            "heart_rate_bpm":     -0.80,   # calm, slow heart rate
            "rmssd":               0.72,   # high HRV = parasympathetic dominance
            "sdnn":                0.65,   # high HRV variability
            "resp_rate_bpm":      -0.40,   # slow, relaxed breathing
            "resp_variability":    0.30,   # regular rhythm
        },
        clinical_notes=[
            "Parasympathetic nervous system dominant",
            "Low sympathetic activation (EDA suppressed)",
            "High HRV indicates relaxed state",
            "Consistent with WESAD baseline condition",
        ],
    ),

    "medium_cognitive_load": DemoProfile(
        key="medium_cognitive_load",
        name="Medium Cognitive Load",
        description="Engaged positive affect state - high arousal but low cognitive demand",
        icon="◈",
        accent="#9B6DFF",
        expected_arousal="High",
        expected_cog_load="Low",
        features={
            "eda_mean":           -0.108,   # Low overall EDA (positive emotion is restful)
            "eda_std":            -0.147,   # Low EDA variability
            "eda_peak_count":     -0.063,   # Few EDA peaks
            "eda_peak_amplitude":  0.328,   # BUT occasional high-amplitude bursts (laughing)
            "heart_rate_bpm":     -0.509,   # Low heart rate (parasympathetic dominance)
            "rmssd":               0.274,   # High HRV (relaxed autonomic state)
            "sdnn":                0.300,   # High HRV variability
            "resp_rate_bpm":      -0.365,   # Low respiration rate (calm)
            "resp_variability":    0.072,   # Regular breathing pattern
        },
        clinical_notes=[
            "HIGH arousal with LOW cognitive load (key distinction from stress)",
            "Parasympathetic dominance despite arousal from engagement/laughter",
            "LOW overall sympathetic activation but with INTERMITTENT peaks",
            "High HRV indicates preserved autonomic flexibility",
            "Physiologically restful yet emotionally active",
            "Consistent with WESAD amusement/comedy condition",
        ],
    ),

    "high_cognitive_load": DemoProfile(
        key="high_cognitive_load",
        name="High Cognitive Load",
        description="Acute cognitive stress - dual-task demand",
        icon="⚡",
        accent="#FF4D7A",
        expected_arousal="High",
        expected_cog_load="High",
        features={
            "eda_mean":            1.65,   # elevated skin conductance
            "eda_std":             1.80,   # volatile EDA signal
            "eda_peak_count":      1.55,   # many stress-related spikes
            "eda_peak_amplitude":  1.70,   # high-amplitude SCRs
            "heart_rate_bpm":      1.90,   # elevated HR
            "rmssd":              -1.50,   # low HRV = sympathetic dominance
            "sdnn":               -1.40,
            "resp_rate_bpm":       1.20,   # rapid breathing
            "resp_variability":   -0.80,
        },
        clinical_notes=[
            "Sympathetic nervous system dominant",
            "Elevated EDA indicates high electrodermal arousal",
            "Suppressed HRV consistent with acute stress",
            "Consistent with WESAD Trier Social Stress Protocol",
        ],
    ),

    "low_emotional_arousal": DemoProfile(
        key="low_emotional_arousal",
        name="Low Emotional Arousal",
        description="Calm, low-activation affective state",
        icon="◎",
        accent="#00FFB2",
        expected_arousal="Low",
        expected_cog_load="Low",
        features={
            "eda_mean":           -0.55,
            "eda_std":            -0.65,
            "eda_peak_count":     -0.60,
            "eda_peak_amplitude": -0.55,
            "heart_rate_bpm":     -0.90,
            "rmssd":               0.80,
            "sdnn":                0.75,
            "resp_rate_bpm":      -0.50,
            "resp_variability":    0.40,
        },
        clinical_notes=[
            "Russell's circumplex: low arousal quadrant",
            "Autonomic balance tilted parasympathetic",
            "EDA quiescence typical of rest/contentment",
            "Mirrors WESAD baseline physiological pattern",
        ],
    ),

    "medium_emotional_arousal": DemoProfile(
        key="medium_emotional_arousal",
        name="Medium Emotional Arousal",
        description="Positive affect with high activation - active engagement state",
        icon="◈",
        accent="#FFB347",
        expected_arousal="High",
        expected_cog_load="Low",
        features={
            "eda_mean":           -0.108,   # Low overall EDA (positive emotion is restful)
            "eda_std":            -0.147,   # Low EDA variability
            "eda_peak_count":     -0.063,   # Few EDA peaks
            "eda_peak_amplitude":  0.328,   # BUT occasional high-amplitude bursts (laughing)
            "heart_rate_bpm":     -0.509,   # Low heart rate (parasympathetic dominance)
            "rmssd":               0.274,   # High HRV (relaxed autonomic state)
            "sdnn":                0.300,   # High HRV variability
            "resp_rate_bpm":      -0.365,   # Low respiration rate (calm)
            "resp_variability":    0.072,   # Regular breathing pattern
        },
        clinical_notes=[
            "Russell's circumplex: high arousal, positive valence quadrant",
            "Sympathetic activation from engagement/entertainment (not threat)",
            "Parasympathetic baseline with engagement-triggered arousal spikes",
            "High HRV characteristic of positive emotional states",
            "Mirrors WESAD amusement/film-viewing condition",
            "Demonstrates dissociation between emotional arousal and cognitive load",
        ],
    ),

    "high_emotional_arousal": DemoProfile(
        key="high_emotional_arousal",
        name="High Emotional Arousal",
        description="High-activation state - acute physiological stress",
        icon="⚡",
        accent="#FF4D4D",
        expected_arousal="High",
        expected_cog_load="High",
        features={
            "eda_mean":            1.75,
            "eda_std":             1.90,
            "eda_peak_count":      1.60,
            "eda_peak_amplitude":  1.80,
            "heart_rate_bpm":      2.00,
            "rmssd":              -1.60,
            "sdnn":               -1.50,
            "resp_rate_bpm":       1.35,
            "resp_variability":   -0.90,
        },
        clinical_notes=[
            "Russell's circumplex: high arousal quadrant",
            "Maximal sympathetic activation across all modalities",
            "EDA, HR and respiration all elevated simultaneously",
            "Mirrors WESAD public speaking stress protocol",
        ],
    ),
}


def get_profile(key: str) -> DemoProfile:
    if key not in PROFILES:
        raise KeyError(f"Unknown profile: {key}. Available: {list(PROFILES)}")
    return PROFILES[key]


def list_profiles() -> list[DemoProfile]:
    return list(PROFILES.values())
