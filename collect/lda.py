from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from dataset_builder import dataset_builder


MODEL_PATH = Path("models/lda_emg.joblib")


def build_model() -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "lda",
                LinearDiscriminantAnalysis(
                    solver="lsqr",
                    shrinkage="auto",
                ),
            ),
        ]
    )


def validate_inputs(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y)

    if X.ndim != 2:
        raise ValueError(
            f"X doit avoir la forme (n_samples, n_features), reçu {X.shape}"
        )

    if y.ndim != 1:
        raise ValueError(f"y doit avoir la forme (n_samples,), reçu {y.shape}")

    if len(X) != len(y):
        raise ValueError(
            f"X et y doivent avoir le même nombre d'échantillons, reçu {len(X)} et {len(y)}"
        )

    if len(np.unique(y)) < 2:
        raise ValueError("Il faut au moins 2 classes pour entraîner un LDA.")

    return X, y


def train_lda(
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[Pipeline, dict[str, Any]]:
    X, y = dataset_builder()
    X, y = validate_inputs(X, y)

    stratify = y if len(np.unique(y)) > 1 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    model = build_model()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "classification_report": classification_report(y_test, y_pred, digits=32),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
    }

    return model, metrics


def save_model(model: Pipeline, path: Path = MODEL_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path: Path = MODEL_PATH) -> Pipeline:
    if not path.exists():
        raise FileNotFoundError(f"Modèle introuvable: {path}")
    return joblib.load(path)


def predict(model: Pipeline, X_new: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    X_new = np.asarray(X_new, dtype=np.float32)

    if X_new.ndim == 1:
        X_new = X_new.reshape(1, -1)

    if X_new.ndim != 2:
        raise ValueError(
            f"X_new doit avoir la forme (n_samples, n_features), reçu {X_new.shape}"
        )

    predictions = model.predict(X_new)
    probabilities = model.predict_proba(X_new)

    return predictions, probabilities


def main() -> None:
    model, metrics = train_lda()

    print("=== Résultats LDA ===")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print("\nClassification report:")
    print(metrics["classification_report"])
    print("Confusion matrix:")
    print(metrics["confusion_matrix"])

    save_model(model)
    print(f"\nModèle sauvegardé dans: {MODEL_PATH}")


if __name__ == "__main__":
    main()