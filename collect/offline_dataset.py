from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"

CLASS_ID_MAP = {
    "Hand_Open": 0,
    "Hand_Close": 1,
    "No_Motion": 2,
    "Wrist_Extension": 3,
    "Wrist_Flexion": 4,
}

CLASS_NAME_MAP = {value: key for key, value in CLASS_ID_MAP.items()}
CLASS_NAME_TO_ID = CLASS_ID_MAP.copy()


@dataclass(frozen=True)
class WindowSample:
    features: np.ndarray
    label: int
    subject_id: str
    source_file: str


def get_class_id_from_file(file_path: Path, class_map: dict[str, int] = CLASS_ID_MAP) -> int | None:
    return class_map.get(file_path.stem)


def load_signal(file_path: Path) -> np.ndarray:
    df = pd.read_csv(file_path, header=None)
    signal = df.values.astype(np.float32)

    if signal.ndim != 2:
        raise ValueError(f"Signal invalide dans {file_path}: shape={signal.shape}")

    return signal


def normalize_signal(signal: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    signal = np.asarray(signal, dtype=np.float32)
    signal = signal - np.mean(signal, axis=0, keepdims=True)

    scale = np.median(np.abs(signal), axis=0, keepdims=True)
    scale = np.where(scale < eps, np.std(signal, axis=0, keepdims=True), scale)
    scale = np.where(scale < eps, 1.0, scale)

    return signal / scale


def trim_signal_edges(signal: np.ndarray, trim_samples: int) -> np.ndarray:
    if trim_samples <= 0:
        return signal

    if signal.shape[0] <= (2 * trim_samples):
        return signal

    return signal[trim_samples:-trim_samples]


def create_windows(signal: np.ndarray, window_size: int, step: int) -> list[np.ndarray]:
    windows: list[np.ndarray] = []
    num_samples = signal.shape[0]

    if num_samples < window_size:
        return windows

    for start in range(0, num_samples - window_size + 1, step):
        end = start + window_size
        windows.append(signal[start:end])

    return windows


def zero_crossings(signal: np.ndarray, threshold: float = 1e-5) -> np.ndarray:
    centered = signal - np.mean(signal, axis=0, keepdims=True)
    sign_changes = np.diff(np.signbit(centered), axis=0)
    magnitude = np.abs(np.diff(centered, axis=0)) >= threshold
    return np.sum(sign_changes & magnitude, axis=0, dtype=np.float32)


def slope_sign_changes(signal: np.ndarray, threshold: float = 1e-5) -> np.ndarray:
    centered = signal - np.mean(signal, axis=0, keepdims=True)
    diff_1 = np.diff(centered, axis=0)
    if diff_1.shape[0] < 2:
        return np.zeros(signal.shape[1], dtype=np.float32)

    prev_diff = diff_1[:-1]
    next_diff = diff_1[1:]
    sign_changes = ((prev_diff > 0) & (next_diff < 0)) | ((prev_diff < 0) & (next_diff > 0))
    magnitude = (np.abs(prev_diff) >= threshold) | (np.abs(next_diff) >= threshold)
    return np.sum(sign_changes & magnitude, axis=0, dtype=np.float32)


def wilson_amplitude(signal: np.ndarray, threshold: float = 0.01) -> np.ndarray:
    diff_signal = np.abs(np.diff(signal, axis=0))
    return np.sum(diff_signal >= threshold, axis=0, dtype=np.float32)


def extract_emg_features(window: np.ndarray) -> np.ndarray:
    window = np.asarray(window, dtype=np.float32)
    abs_window = np.abs(window)
    diff_window = np.diff(window, axis=0)

    mav = np.mean(abs_window, axis=0)
    iemg = np.sum(abs_window, axis=0)
    rms = np.sqrt(np.mean(window ** 2, axis=0))
    variance = np.var(window, axis=0)
    log_detector = np.exp(np.mean(np.log(abs_window + 1e-6), axis=0))
    wl = np.sum(np.abs(diff_window), axis=0)
    dasdv = np.sqrt(np.mean(diff_window ** 2, axis=0))
    aac = np.mean(np.abs(diff_window), axis=0)
    zc = zero_crossings(window)
    ssc = slope_sign_changes(window)
    wamp = wilson_amplitude(window)

    features = np.concatenate(
        [
            mav,
            iemg,
            rms,
            variance,
            log_detector,
            wl,
            dasdv,
            aac,
            zc,
            ssc,
            wamp,
        ]
    ).astype(np.float32)
    return features


def iter_subject_files(data_dir: Path = DATA_DIR) -> list[tuple[str, Path]]:
    subject_files: list[tuple[str, Path]] = []

    if not data_dir.exists():
        raise FileNotFoundError(f"Dossier de donnees introuvable: {data_dir}")

    for subject_dir in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        for csv_file in sorted(subject_dir.glob("*.csv")):
            subject_files.append((subject_dir.name, csv_file))

    if not subject_files:
        raise FileNotFoundError(f"Aucun fichier CSV trouve dans {data_dir}")

    return subject_files


def build_feature_dataset(
    fs_emg: int = 2000,
    window_size_ms: int = 200,
    step_ms: int = 50,
    trim_edges_ms: int = 250,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    window_size = int(window_size_ms * fs_emg / 1000)
    step = int(step_ms * fs_emg / 1000)
    trim_samples = int(trim_edges_ms * fs_emg / 1000)

    samples: list[WindowSample] = []

    for subject_id, file_path in iter_subject_files():
        class_id = get_class_id_from_file(file_path)
        if class_id is None:
            print(f"Ignore (classe inconnue): {file_path.name}")
            continue

        signal = load_signal(file_path)
        signal = trim_signal_edges(signal, trim_samples=trim_samples)
        signal = normalize_signal(signal)
        windows = create_windows(signal, window_size, step)

        for window in windows:
            samples.append(
                WindowSample(
                    features=extract_emg_features(window),
                    label=class_id,
                    subject_id=subject_id,
                    source_file=file_path.stem,
                )
            )

    if not samples:
        raise ValueError("Aucune fenetre exploitable n'a ete construite a partir des donnees.")

    X = np.stack([sample.features for sample in samples]).astype(np.float32)
    y = np.asarray([sample.label for sample in samples], dtype=np.int64)
    subjects = np.asarray([sample.subject_id for sample in samples])
    files = np.asarray([f"{sample.subject_id}/{sample.source_file}" for sample in samples])

    print("X_features:", X.shape)
    print("y:", y.shape)

    unique, counts = np.unique(y, return_counts=True)
    for class_id, count in zip(unique, counts):
        print(f"Classe {class_id} ({CLASS_NAME_MAP[class_id]}): {count} fenetres")

    unique_subjects = np.unique(subjects)
    print(f"Sujets: {len(unique_subjects)} -> {', '.join(unique_subjects)}")

    return X, y, subjects, files


def filter_dataset_by_classes(
    X: np.ndarray,
    y: np.ndarray,
    subjects: np.ndarray,
    files: np.ndarray,
    class_names: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[int, str]]:
    selected_ids: list[int] = []
    for class_name in class_names:
        if class_name not in CLASS_NAME_TO_ID:
            raise ValueError(f"Classe inconnue: {class_name}")
        selected_ids.append(CLASS_NAME_TO_ID[class_name])

    mask = np.isin(y, selected_ids)
    if not np.any(mask):
        raise ValueError("Aucun echantillon ne correspond aux classes demandees.")

    X_filtered = X[mask]
    y_filtered = y[mask]
    subjects_filtered = subjects[mask]
    files_filtered = files[mask]

    remap = {old_id: new_id for new_id, old_id in enumerate(selected_ids)}
    y_remapped = np.asarray([remap[int(label)] for label in y_filtered], dtype=np.int64)
    remapped_name_map = {new_id: CLASS_NAME_MAP[old_id] for old_id, new_id in remap.items()}

    return X_filtered, y_remapped, subjects_filtered, files_filtered, remapped_name_map
