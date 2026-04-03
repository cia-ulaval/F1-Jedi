from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from lda_offline import (
    build_svm_model,
    evaluate_leave_one_subject_out,
    evaluate_model_candidates,
    group_split_preserves_classes,
    save_offline_plots,
    split_by_groups,
    split_stratified_windows,
    validate_inputs,
)
from offline_dataset import build_feature_dataset


REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = REPO_ROOT / "models" / "svm_emg_features.joblib"


def train_svm_offline(
    X: np.ndarray | None = None,
    y: np.ndarray | None = None,
    sessions: np.ndarray | None = None,
    files: np.ndarray | None = None,
    validation_mode: str = "auto",
    test_size: float = 0.2,
    random_state: int = 42,
    probability: bool = True,
) -> tuple[Any, dict[str, Any]]:
    if X is None or y is None or sessions is None or files is None:
        X, y, sessions, files = build_feature_dataset()

    if validation_mode not in {"auto", "session", "subject", "file"}:
        raise ValueError("validation_mode doit etre 'auto', 'session', 'subject' ou 'file'.")

    group_candidates = {
        "session": sessions,
        "subject": sessions,
        "file": files,
    }

    if validation_mode == "auto":
        chosen_mode = "session" if len(np.unique(sessions)) >= 2 else "file"
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
            "mais moins fiable qu'une vraie validation par session."
        )

    model = build_svm_model(probability=probability)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    metrics.update(
        {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "classification_report": classification_report(y_test, y_pred, digits=4, zero_division=0),
            "confusion_matrix": confusion_matrix(y_test, y_pred),
            "n_train": int(len(y_train)),
            "n_test": int(len(y_test)),
            "y_test": y_test,
            "y_pred": y_pred,
        }
    )

    return model, metrics


def save_model(model: Any, path: Path = MODEL_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entrainement offline d'un SVM EMG.")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Entraine et sauvegarde le modele sans lancer le LOSO ni la comparaison complete.",
    )
    parser.add_argument(
        "--proba",
        action="store_true",
        help="Active predict_proba. Plus lent a l'entrainement.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    X, y, sessions, files = build_feature_dataset()
    model, metrics = train_svm_offline(
        X=X,
        y=y,
        sessions=sessions,
        files=files,
        probability=args.proba,
    )
    loso_metrics = None
    model_comparison: list[dict[str, Any]] = []

    if not args.quick:
        loso_metrics = (
            evaluate_leave_one_subject_out(
                X,
                y,
                sessions,
                model_builder=lambda: build_svm_model(probability=args.proba),
            )
            if len(np.unique(sessions)) >= 2
            else None
        )
        model_comparison = (
            evaluate_model_candidates(X, y, sessions)
            if len(np.unique(sessions)) >= 2
            else []
        )

    print("=== Resultats SVM Offline (features EMG) ===")
    validation_mode_label = "session" if metrics["validation_mode"] in {"session", "subject"} else metrics["validation_mode"]
    print(f"Mode rapide: {'oui' if args.quick else 'non'}")
    print(f"Probabilites: {'oui' if args.proba else 'non'}")
    print(f"Validation: {validation_mode_label}")
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
        print("\n=== Leave-One-Session-Out ===")
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
        print("\n=== Comparaison Modeles (LOSO par session) ===")
        for result in model_comparison:
            print(f"{result['model_name']}: {result['mean_accuracy']:.4f}")

    plot_paths = save_offline_plots(
        metrics,
        loso_metrics,
        model_comparison,
        X,
        y,
        sessions,
        plot_prefix="svm",
        plot_label="SVM Offline",
    )

    save_model(model)
    print(f"\nModele sauvegarde dans: {MODEL_PATH}")
    if plot_paths:
        print("Graphiques sauvegardes dans:")
        for path in plot_paths:
            print(path)


if __name__ == "__main__":
    main()
