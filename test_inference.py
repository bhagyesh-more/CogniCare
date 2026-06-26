from src.prediction_engine import PredictionEngine

engine = PredictionEngine()
engine.load()

sample = {
    "eda_mean": 0.5,
    "eda_std": 0.2,
    "eda_peak_count": 15,
    "eda_peak_amplitude": 0.3,
    "heart_rate_bpm": 75,
    "rmssd": 40,
    "sdnn": 55,
    "resp_rate_bpm": 14,
    "resp_variability": 2.1
}

print("Emotional Arousal")
print(engine.predict_emotional_arousal(sample))

print("\nCognitive Load")
print(engine.predict_cognitive_load(sample))