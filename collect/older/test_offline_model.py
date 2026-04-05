from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from offline_dataset import (
    CLASS_NAME_MAP,
    create_windows,
    extract_emg_features,
    get_class_id_from_file,
    load_signal,
    normalize_signal,
    trim_signal_edges,
    DEFAULT_FS_EMG,
    DEFAULT_STEP_MS,
    DEFAULT_TRIM_EDGES_MS,
    DEFAULT_WINDOW_SIZE_MS,
)
from offline_model import load_model, predict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Charge un modele offline et predit sur un vrai fichier CSV ou sur une entree factice."
    )
    parser.add_argument("model_name", nargs="?", default="svm", choices=["lda", "svm"])
    parser.add_argument("csv_path", nargs="?", default=None)
    return parser.parse_args()


def build_features_from_csv(
    csv_path: Path,
    fs_emg: int = DEFAULT_FS_EMG,
    window_size_ms: int = DEFAULT_WINDOW_SIZE_MS,
    step_ms: int = DEFAULT_STEP_MS,
    trim_edges_ms: int = DEFAULT_TRIM_EDGES_MS,
) -> np.ndarray:
    signal = load_signal(csv_path)
    trim_samples = int(trim_edges_ms * fs_emg / 1000)
    window_size = int(window_size_ms * fs_emg / 1000)
    step = int(step_ms * fs_emg / 1000)

    signal = trim_signal_edges(signal, trim_samples=trim_samples)
    signal = normalize_signal(signal)
    windows = create_windows(signal, window_size, step)

    if not windows:
        raise ValueError(
            f"Aucune fenetre exploitable pour {csv_path}. Le fichier est peut-etre trop court."
        )

    return np.stack([extract_emg_features(window) for window in windows]).astype(np.float32)


def main() -> None:
    args = parse_args()
    model = load_model(model_name=args.model_name)

    print("Modele:", args.model_name)

    if args.csv_path is None:
        feature_count = int(getattr(model, "n_features_in_", 11))
        x_new = np.zeros(feature_count, dtype=np.float32)
        pred, proba = predict(model, x_new)
        print("Mode test: entree factice")
        print("Classe predite:", pred)
        print("Probabilites:", proba)
        return

    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Fichier introuvable: {csv_path}")

    X_new = build_features_from_csv(csv_path)
    pred, proba = predict(model, X_new)

    pred = np.asarray(pred, dtype=np.int64)
    majority_class, majority_count = Counter(pred.tolist()).most_common(1)[0]
    majority_ratio = majority_count / len(pred)

    expected_class_id = get_class_id_from_file(csv_path)
    expected_class_name = (
        CLASS_NAME_MAP[expected_class_id] if expected_class_id is not None else "inconnue"
    )

    print(f"Fichier teste: {csv_path}")
    print(f"Classe attendue (depuis le nom du fichier): {expected_class_name}")
    print(f"Nombre de fenetres: {len(pred)}")
    print(
        "Classe majoritaire predite: "
        f"{majority_class} ({CLASS_NAME_MAP.get(int(majority_class), 'inconnue')})"
    )
    print(f"Confiance majoritaire par vote: {majority_ratio:.4f}")

    distribution = Counter(pred.tolist())
    print("Repartition des predictions par fenetre:")
    for class_id, count in sorted(distribution.items()):
        class_name = CLASS_NAME_MAP.get(int(class_id), "inconnue")
        print(f"- {class_id} ({class_name}): {count}")

    if proba is None:
        print("Probabilites: indisponibles pour ce modele.")
    else:
        mean_proba = np.mean(proba, axis=0)
        print("Probabilites moyennes sur toutes les fenetres:")
        for class_id, value in enumerate(mean_proba):
            class_name = CLASS_NAME_MAP.get(class_id, "inconnue")
            print(f"- {class_id} ({class_name}): {value:.4f}")


if __name__ == "__main__":
    main()
