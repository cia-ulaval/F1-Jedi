from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATHS = {
    "lda": REPO_ROOT / "models" / "lda_emg_features.joblib",
    "svm": REPO_ROOT / "models" / "svm_emg_features.joblib",
}


def get_model_path(model_name: str = "svm") -> Path:
    normalized_name = model_name.strip().lower()
    if normalized_name not in MODEL_PATHS:
        available = ", ".join(sorted(MODEL_PATHS))
        raise ValueError(f"Modele inconnu: {model_name}. Modeles disponibles: {available}")
    return MODEL_PATHS[normalized_name]


def load_model(model_name: str = "svm", path: Path | None = None):
    model_path = path if path is not None else get_model_path(model_name)
    if not model_path.exists():
        raise FileNotFoundError(f"Modele introuvable: {model_path}")
    return joblib.load(model_path)


def predict(model, X_new: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    X_new = np.asarray(X_new, dtype=np.float32)

    if X_new.ndim == 1:
        X_new = X_new.reshape(1, -1)

    if X_new.ndim != 2:
        raise ValueError(
            f"X_new doit avoir la forme (n_samples, n_features), recu {X_new.shape}"
        )

    predictions = model.predict(X_new)
    probabilities = model.predict_proba(X_new) if hasattr(model, "predict_proba") else None

    return predictions, probabilities
