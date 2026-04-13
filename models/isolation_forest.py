from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


class IsolationForestDetector:
    """Isolation Forest anomaly detector for windowed climate sequences."""

    def __init__(self, config) -> None:
        self.config = config
        self.model: IsolationForest | None = None
        self.scaler = StandardScaler()
        self.used_n_jobs = config.N_JOBS

    @staticmethod
    def _flatten_windows(X: np.ndarray) -> np.ndarray:
        return X.reshape(X.shape[0], -1)

    def fit(self, X_train: np.ndarray) -> "IsolationForestDetector":
        X_train_flat = self._flatten_windows(X_train)
        X_train_scaled = self.scaler.fit_transform(X_train_flat)
        max_samples = min(self.config.MAX_SAMPLES, len(X_train_scaled))

        def build_model(n_jobs: int) -> IsolationForest:
            return IsolationForest(
                n_estimators=self.config.N_ESTIMATORS,
                contamination=self.config.CONTAMINATION,
                max_samples=max_samples,
                max_features=self.config.MAX_FEATURES,
                bootstrap=self.config.BOOTSTRAP,
                random_state=self.config.RANDOM_STATE,
                n_jobs=n_jobs,
            )

        self.used_n_jobs = self.config.N_JOBS
        self.model = build_model(self.used_n_jobs)
        try:
            self.model.fit(X_train_scaled)
        except PermissionError:
            self.used_n_jobs = 1
            self.model = build_model(self.used_n_jobs)
            self.model.fit(X_train_scaled)
        return self

    def detect(self, X: np.ndarray) -> dict[str, np.ndarray]:
        if self.model is None:
            raise RuntimeError("Isolation Forest model has not been fitted yet.")

        X_flat = self._flatten_windows(X)
        X_scaled = self.scaler.transform(X_flat)
        predictions = self.model.predict(X_scaled)
        raw_scores = -self.model.decision_function(X_scaled)

        score_range = raw_scores.max() - raw_scores.min()
        if score_range > 0:
            anomaly_scores = (raw_scores - raw_scores.min()) / score_range
        else:
            anomaly_scores = np.zeros_like(raw_scores)

        return {
            "anomalies": predictions == -1,
            "anomaly_scores": anomaly_scores,
            "raw_scores": raw_scores,
        }

    def fit_and_detect(self, X_train: np.ndarray, X_test: np.ndarray) -> dict[str, np.ndarray]:
        self.fit(X_train)
        return self.detect(X_test)

    def save(self, output_path: Path) -> None:
        if self.model is None:
            raise RuntimeError("Cannot save an unfitted Isolation Forest model.")

        with output_path.open("wb") as handle:
            pickle.dump({"model": self.model, "scaler": self.scaler}, handle)
