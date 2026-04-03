from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit, LeaveOneGroupOut, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from offline_dataset import build_feature_dataset
from offline_dataset import CLASS_NAME_MAP


REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = REPO_ROOT / "models" / "lda_emg_features.joblib"
PLOTS_DIR = REPO_ROOT / "models" / "plots"


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


def build_svm_model(probability: bool = True) -> Pipeline:
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
                    probability=probability,
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


def _class_labels(y: np.ndarray) -> list[str]:
    return [CLASS_NAME_MAP[int(class_id)] for class_id in sorted(np.unique(y))]


def plot_confusion_matrix_figure(
    cm: np.ndarray,
    labels: list[str],
    title: str,
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_title(title)
    ax.set_xlabel("Prediction")
    ax.set_ylabel("Verite")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_yticklabels(labels)

    max_value = float(np.max(cm)) if cm.size else 0.0
    threshold = max_value / 2.0 if max_value > 0 else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = int(cm[i, j])
            ax.text(
                j,
                i,
                str(value),
                ha="center",
                va="center",
                color="white" if value > threshold else "black",
            )

    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_loso_fold_accuracies(folds: list[dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sessions = [",".join(fold["test_groups"]) for fold in folds]
    accuracies = [fold["accuracy"] for fold in folds]

    fig, ax = plt.subplots(figsize=(12, 5))
    colors = ["#1f77b4" if acc >= np.mean(accuracies) else "#d62728" for acc in accuracies]
    ax.bar(sessions, accuracies, color=colors)
    ax.axhline(np.mean(accuracies), color="black", linestyle="--", linewidth=1.2, label="Moyenne LOSO")
    ax.set_title("Accuracy LOSO par session")
    ax.set_xlabel("Session testee")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.0, max(1.0, max(accuracies) + 0.05))
    ax.tick_params(axis="x", rotation=45)
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_model_comparison(results: list[dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    names = [result["model_name"] for result in results]
    scores = [result["mean_accuracy"] for result in results]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(names, scores, color=["#2ca02c", "#1f77b4", "#ff7f0e"][: len(names)])
    ax.set_title("Comparaison des modeles en LOSO")
    ax.set_xlabel("Modele")
    ax.set_ylabel("Accuracy moyenne")
    ax.set_ylim(0.0, max(1.0, max(scores) + 0.05))

    for idx, score in enumerate(scores):
        ax.text(idx, score + 0.01, f"{score:.3f}", ha="center", va="bottom")

    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_feature_projection(
    X: np.ndarray,
    y: np.ndarray,
    sessions: np.ndarray,
    out_path: Path,
    max_points: int = 2500,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if len(X) > max_points:
        rng = np.random.default_rng(42)
        keep_idx = rng.choice(len(X), size=max_points, replace=False)
        X_plot = X[keep_idx]
        y_plot = y[keep_idx]
        sessions_plot = sessions[keep_idx]
    else:
        X_plot = X
        y_plot = y
        sessions_plot = sessions

    X_scaled = StandardScaler().fit_transform(X_plot)
    coords = PCA(n_components=2, random_state=42).fit_transform(X_scaled)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    class_labels = _class_labels(y)
    cmap_classes = plt.get_cmap("tab10")
    cmap_sessions = plt.get_cmap("tab20")

    for class_id in sorted(np.unique(y_plot)):
        mask = y_plot == class_id
        axes[0].scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=12,
            alpha=0.6,
            label=CLASS_NAME_MAP[int(class_id)],
            color=cmap_classes(int(class_id) % 10),
        )
    axes[0].set_title("Projection PCA coloree par classe")
    axes[0].set_xlabel("PC1")
    axes[0].set_ylabel("PC2")
    axes[0].legend(fontsize=8, loc="best")

    unique_sessions = np.unique(sessions_plot)
    for idx, session in enumerate(unique_sessions):
        mask = sessions_plot == session
        axes[1].scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=12,
            alpha=0.6,
            label=session,
            color=cmap_sessions(idx % 20),
        )
    axes[1].set_title("Projection PCA coloree par session")
    axes[1].set_xlabel("PC1")
    axes[1].set_ylabel("PC2")
    if len(unique_sessions) <= 20:
        axes[1].legend(fontsize=7, loc="best", ncol=2)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_offline_plots(
    metrics: dict[str, Any],
    loso_metrics: dict[str, Any] | None,
    model_comparison: list[dict[str, Any]],
    X: np.ndarray,
    y: np.ndarray,
    sessions: np.ndarray,
    plot_prefix: str = "lda",
    plot_label: str = "LDA Offline",
) -> list[Path]:
    labels = _class_labels(y)
    saved_paths: list[Path] = []

    split_cm_path = PLOTS_DIR / f"{plot_prefix}_session_split_confusion_matrix.png"
    plot_confusion_matrix_figure(
        metrics["confusion_matrix"],
        labels,
        f"{plot_label} - Confusion Matrix (split principal par session)",
        split_cm_path,
    )
    saved_paths.append(split_cm_path)

    if loso_metrics is not None:
        loso_cm_path = PLOTS_DIR / f"{plot_prefix}_loso_confusion_matrix.png"
        plot_confusion_matrix_figure(
            loso_metrics["confusion_matrix"],
            labels,
            f"{plot_label} - Confusion Matrix LOSO",
            loso_cm_path,
        )
        saved_paths.append(loso_cm_path)

        loso_bar_path = PLOTS_DIR / f"{plot_prefix}_loso_accuracy_per_session.png"
        plot_loso_fold_accuracies(loso_metrics["folds"], loso_bar_path)
        saved_paths.append(loso_bar_path)

    if model_comparison:
        model_cmp_path = PLOTS_DIR / f"{plot_prefix}_model_comparison_loso.png"
        plot_model_comparison(model_comparison, model_cmp_path)
        saved_paths.append(model_cmp_path)

    projection_path = PLOTS_DIR / f"{plot_prefix}_feature_projection_pca.png"
    plot_feature_projection(X, y, sessions, projection_path)
    saved_paths.append(projection_path)

    return saved_paths


def train_lda_offline(
    X: np.ndarray | None = None,
    y: np.ndarray | None = None,
    sessions: np.ndarray | None = None,
    files: np.ndarray | None = None,
    validation_mode: str = "auto",
    test_size: float = 0.2,
    random_state: int = 42,
    subjects: np.ndarray | None = None,
) -> tuple[Pipeline, dict[str, Any]]:
    if sessions is None and subjects is not None:
        sessions = subjects

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
            "y_test": y_test,
            "y_pred": y_pred,
        }
    )

    return model, metrics


def evaluate_leave_one_subject_out(
    X: np.ndarray,
    y: np.ndarray,
    sessions: np.ndarray,
    model_builder: Any,
) -> dict[str, Any]:
    logo = LeaveOneGroupOut()
    fold_metrics: list[dict[str, Any]] = []
    aggregated_true: list[np.ndarray] = []
    aggregated_pred: list[np.ndarray] = []

    for train_idx, test_idx in logo.split(X, y, groups=sessions):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        train_sessions = np.unique(sessions[train_idx])
        test_sessions = np.unique(sessions[test_idx])

        model = model_builder()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        aggregated_true.append(y_test)
        aggregated_pred.append(y_pred)
        fold_metrics.append(
            {
                "train_groups": train_sessions,
                "test_groups": test_sessions,
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
    sessions: np.ndarray,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for model_name, model_builder in get_model_builders().items():
        metrics = evaluate_leave_one_subject_out(X, y, sessions, model_builder=model_builder)
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
    X, y, sessions, files = build_feature_dataset()
    model, metrics = train_lda_offline(X=X, y=y, sessions=sessions, files=files)
    loso_metrics = (
        evaluate_leave_one_subject_out(X, y, sessions, model_builder=build_lda_model)
        if len(np.unique(sessions)) >= 2
        else None
    )
    model_comparison = (
        evaluate_model_candidates(X, y, sessions)
        if len(np.unique(sessions)) >= 2
        else []
    )

    print("=== Resultats LDA Offline (features EMG) ===")
    validation_mode_label = "session" if metrics["validation_mode"] in {"session", "subject"} else metrics["validation_mode"]
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

    plot_paths = save_offline_plots(metrics, loso_metrics, model_comparison, X, y, sessions)

    save_model(model)
    print(f"\nModele sauvegarde dans: {MODEL_PATH}")
    print("Graphiques sauvegardes dans:")
    for path in plot_paths:
        print(path)


if __name__ == "__main__":
    main()
