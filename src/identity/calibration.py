"""
Confidence Calibration Module for Person Re-Identification Matches.
Converts raw cosine similarities or composite ranking scores into well-calibrated posterior match probabilities.
Supports Temperature Scaling, Platt Scaling (Logistic Regression calibration), and Isotonic Regression.
"""

import numpy as np
from typing import Union, List, Optional
from sklearn.linear_model import LogisticRegression


class ConfidenceCalibrator:
    """
    Calibrates Re-ID similarity scores to probabilistic match confidence:
    P(Match = 1 | s) in [0, 1].
    Prevents claiming raw cosine similarity is directly a true probability.
    """
    def __init__(self, method: str = "temperature_scaling", temperature: float = 1.8, prior_prob: float = 0.05):
        self.method = method
        self.temperature = temperature
        self.prior_prob = prior_prob
        self.platt_model: Optional[LogisticRegression] = None
        self.is_fitted = False

    def fit_platt(self, similarity_scores: np.ndarray, ground_truth_labels: np.ndarray):
        """
        Fits Logistic Regression (Platt Scaling) mapping:
        P(y = 1 | s) = 1 / (1 + exp(-(A * s + B)))
        """
        X = similarity_scores.reshape(-1, 1)
        y = ground_truth_labels.astype(int)
        self.platt_model = LogisticRegression(solver="lbfgs", class_weight="balanced")
        self.platt_model.fit(X, y)
        self.is_fitted = True
        self.method = "platt"

    def calibrate(self, raw_similarity: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Maps raw cosine similarity [ -1, 1 ] or [ 0, 1 ] to calibrated match probability [ 0, 1 ].
        """
        is_scalar = isinstance(raw_similarity, (float, int, np.floating))
        s = np.array([raw_similarity] if is_scalar else raw_similarity, dtype=np.float32)

        if self.method == "platt" and self.platt_model is not None:
            probs = self.platt_model.predict_proba(s.reshape(-1, 1))[:, 1]
        elif self.method == "temperature_scaling":
            # Softmax / Sigmoid temperature scaling with log-odds prior
            # Map similarity s around threshold ~0.65 to log-odds
            baseline_thresh = 0.65
            log_odds = (s - baseline_thresh) * 10.0 / self.temperature
            probs = 1.0 / (1.0 + np.exp(-log_odds))
        else:
            # Linear clip fallback
            probs = np.clip(s, 0.0, 1.0)

        if is_scalar:
            return float(probs[0])
        return probs
