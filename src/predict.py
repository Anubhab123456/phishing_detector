"""Inference + local (per-URL) explanation."""
import json
import os

import joblib
import numpy as np

from src.features import FEATURE_NAMES, extract_features, features_to_vector

MODEL_PATH = "models/model.joblib"
METRICS_PATH = "models/metrics.json"


class PhishingDetector:
    def __init__(self, model_path: str = MODEL_PATH):
        if not os.path.exists(model_path):
            raise FileNotFoundError("Model not found. Run:  python -m src.train")
        b = joblib.load(model_path)
        self.model = b["model"]
        self.model_name = b["model_name"]
        self.baseline = b["baseline"]

    def predict(self, url: str) -> dict:
        x = features_to_vector(url).reshape(1, -1)
        p_phish = float(self.model.predict_proba(x)[0, 1])
        label = "phishing" if p_phish >= 0.5 else "legitimate"
        confidence = p_phish if label == "phishing" else 1 - p_phish
        return {"url": url, "label": label, "phishing_probability": p_phish,
                "confidence": confidence, "features": extract_features(url),
                "contributions": self._explain(x, p_phish)}

    def _explain(self, x: np.ndarray, p_phish: float) -> dict:
        """Occlusion-style local explanation: replace each feature with the typical
        legitimate value and measure the change in phishing probability.
        Positive value => the feature pushed the prediction toward PHISHING."""
        X_mod = np.repeat(x, len(FEATURE_NAMES), axis=0)
        for i in range(len(FEATURE_NAMES)):
            X_mod[i, i] = self.baseline[i]
        p_mod = self.model.predict_proba(X_mod)[:, 1]
        contrib = p_phish - p_mod
        return dict(sorted(zip(FEATURE_NAMES, contrib.tolist()), key=lambda kv: -abs(kv[1])))


def load_metrics(path: str = METRICS_PATH):
    with open(path) as f:
        return json.load(f)
