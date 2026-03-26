from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit, LeaveOneGroupOut, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from offline_dataset import build_feature_dataset


REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = REPO_ROOT / "models" / "lda_emg_features.joblib"


def build_lda_model() -> Pipeline:
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


def build_svm_model() -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "svm",
                SVC(
                    kernel="rbf",
                    C=3.0,
                    gamma="scale",
                    class_weight="balanced",
                ),
            ),
        ]
    )


def build_random_forest_model() -> Pipeline:
    return Pipeline(
        steps=[
            (
                "rf",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=None,
                    min_samples_leaf=2,
                    random_state=42,
                    class_weight="balanced_subsample",
                    n_jobs=1,
                ),
            ),
        ]
    )


def get_model_builders() -> dict[str, Any]:
    return {
        "lda": build_lda_model,
        "svm": build_svm_model,
        "random_forest": build_random_forest_model,
    }

def validate_inputs(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y)
    groups = np.asarray(groups)

    if X.ndim != 2:
        raise ValueError(f"X doit avoir la forme (n_samples, n_features), recu {X.shape}")

    if y.ndim != 1:
        raise ValueError(f"y doit avoir la forme (n_samples,), recu {y.shape}")

    if groups.ndim != 1:
        raise ValueError(f"groups doit avoir la forme (n_samples,), recu {groups.shape}")

    if len(X) != len(y) or len(X) != len(groups):
        raise ValueError("X, y et groups doivent avoir le meme nombre d'echantillons.")

    if len(np.unique(y)) < 2:
        raise ValueError("Il faut au moins 2 classes pour entrainer un LDA.")

    if len(np.unique(groups)) < 2:
        raise ValueError(
            "La validation par groupe demande au moins 2 groupes distincts."
        )

    return X, y, groups


def split_by_groups(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))

    return (
        X[train_idx],
        X[test_idx],
        y[train_idx],
        y[test_idx],
        groups[train_idx],
        groups[test_idx],
    )


def split_stratified_windows(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    return X_train, X_test, y_train, y_test


def group_split_preserves_classes(
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> bool:
    train_classes = set(np.unique(y_train))
    test_classes = set(np.unique(y_test))
    return train_classes == test_classes


def train_lda_offline(
    validation_mode: str = "auto",
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[Pipeline, dict[str, Any]]:
    X, y, subjects, files = build_feature_dataset()

    if validation_mode not in {"auto", "subject", "file"}:
        raise ValueError("validation_mode doit etre 'auto', 'subject' ou 'file'.")

    group_candidates = {
        "subject": subjects,
        "file": files,
    }

    if validation_mode == "auto":
        chosen_mode = "subject" if len(np.unique(subjects)) >= 2 else "file"
    else:
        chosen_mode = validation_mode

    metrics: dict[str, Any] = {}

    try:
        groups = group_candidates[chosen_mode]
        X, y, groups = validate_inputs(X, y, groups)
        X_train, X_test, y_train, y_test, train_groups, test_groups = split_by_groups(
            X,
            y,
            groups,
            test_size=test_size,
            random_state=random_state,
        )

        if not group_split_preserves_classes(y_train, y_test):
            raise ValueError(
                "Le split groupe retire au moins une classe complete du train ou du test."
            )

        metrics["validation_mode"] = chosen_mode
        metrics["train_groups"] = np.unique(train_groups)
        metrics["test_groups"] = np.unique(test_groups)
        metrics["validation_warning"] = ""
    except ValueError as exc:
        X, y = np.asarray(X, dtype=np.float32), np.asarray(y)
        X_train, X_test, y_train, y_test = split_stratified_windows(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
        )
        metrics["validation_mode"] = "window_stratified_fallback"
        metrics["train_groups"] = np.array([], dtype=str)
        metrics["test_groups"] = np.array([], dtype=str)
        metrics["validation_warning"] = (
            "Validation groupee impossible avec le dataset actuel. "
            f"Raison: {exc} "
            "Le script a bascule sur un split stratifie par fenetres, utile pour du debug "
            "mais moins fiable qu'une vraie validation par sujet/session."
        )

    model = build_lda_model()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    metrics.update(
        {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "classification_report": classification_report(y_test, y_pred, digits=4, zero_division=0),
            "confusion_matrix": confusion_matrix(y_test, y_pred),
            "n_train": int(len(y_train)),
            "n_test": int(len(y_test)),
        }
    )

    return model, metrics


def evaluate_leave_one_subject_out(
    X: np.ndarray,
    y: np.ndarray,
    subjects: np.ndarray,
    model_builder: Any,
) -> dict[str, Any]:
    logo = LeaveOneGroupOut()
    fold_metrics: list[dict[str, Any]] = []
    aggregated_true: list[np.ndarray] = []
    aggregated_pred: list[np.ndarray] = []

    for train_idx, test_idx in logo.split(X, y, groups=subjects):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        train_subjects = np.unique(subjects[train_idx])
        test_subjects = np.unique(subjects[test_idx])

        model = model_builder()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        aggregated_true.append(y_test)
        aggregated_pred.append(y_pred)
        fold_metrics.append(
            {
                "train_groups": train_subjects,
                "test_groups": test_subjects,
                "accuracy": float(accuracy_score(y_test, y_pred)),
            }
        )

    y_all_true = np.concatenate(aggregated_true)
    y_all_pred = np.concatenate(aggregated_pred)

    return {
        "folds": fold_metrics,
        "mean_accuracy": float(np.mean([fold["accuracy"] for fold in fold_metrics])),
        "classification_report": classification_report(
            y_all_true,
            y_all_pred,
            digits=4,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(y_all_true, y_all_pred),
    }


def evaluate_model_candidates(
    X: np.ndarray,
    y: np.ndarray,
    subjects: np.ndarray,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for model_name, model_builder in get_model_builders().items():
        metrics = evaluate_leave_one_subject_out(X, y, subjects, model_builder=model_builder)
        results.append(
            {
                "model_name": model_name,
                "mean_accuracy": metrics["mean_accuracy"],
                "folds": metrics["folds"],
            }
        )

    results.sort(key=lambda item: item["mean_accuracy"], reverse=True)
    return results


def save_model(model: Pipeline, path: Path = MODEL_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def main() -> None:
    model, metrics = train_lda_offline()
    X, y, subjects, _ = build_feature_dataset()
    loso_metrics = (
        evaluate_leave_one_subject_out(X, y, subjects, model_builder=build_lda_model)
        if len(np.unique(subjects)) >= 2
        else None
    )
    model_comparison = (
        evaluate_model_candidates(X, y, subjects)
        if len(np.unique(subjects)) >= 2
        else []
    )

    print("=== Resultats LDA Offline (features EMG) ===")
    print(f"Validation: {metrics['validation_mode']}")
    if metrics["validation_warning"]:
        print(f"Warning: {metrics['validation_warning']}")
    if len(metrics["train_groups"]) > 0:
        print(f"Train groups: {', '.join(metrics['train_groups'])}")
    if len(metrics["test_groups"]) > 0:
        print(f"Test groups: {', '.join(metrics['test_groups'])}")
    print(f"Echantillons train: {metrics['n_train']}")
    print(f"Echantillons test: {metrics['n_test']}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print("\nClassification report:")
    print(metrics["classification_report"])
    print("Confusion matrix:")
    print(metrics["confusion_matrix"])

    if loso_metrics is not None:
        print("\n=== Leave-One-Subject-Out ===")
        print(f"Accuracy moyenne: {loso_metrics['mean_accuracy']:.4f}")
        for index, fold in enumerate(loso_metrics["folds"], start=1):
            print(
                f"Fold {index}: train={', '.join(fold['train_groups'])} "
                f"test={', '.join(fold['test_groups'])} "
                f"accuracy={fold['accuracy']:.4f}"
            )
        print("\nClassification report LOSO:")
        print(loso_metrics["classification_report"])
        print("Confusion matrix LOSO:")
        print(loso_metrics["confusion_matrix"])

    if model_comparison:
        print("\n=== Comparaison Modeles (LOSO) ===")
        for result in model_comparison:
            print(f"{result['model_name']}: {result['mean_accuracy']:.4f}")

    save_model(model)
    print(f"\nModele sauvegarde dans: {MODEL_PATH}")


if __name__ == "__main__":
    main()
