import numpy as np
import pandas as pd
from pathlib import Path
import re


def dataset_builder(fs_emg=2000, window_size_ms=200, step_ms=20):
    data_dir = Path("../data")

    window_size = int(window_size_ms * fs_emg / 1000)
    step = int(step_ms * fs_emg / 1000)

    all_windows = []
    all_labels = []

    # trouver les fichiers EMG
    emg_files = sorted(data_dir.rglob("*emg*.csv"))

    for file in emg_files:
        # extraire class_id depuis le nom du fichier
        match_class_id = re.search(r"C(\d+)", file.name)
        class_id = int(match_class_id.group(1)) if match_class_id else 0

        # charger le csv
        data_frame = pd.read_csv(file)
        signal = data_frame.values

        num_of_samples = signal.shape[0]

        # fenetrage
        for start in range(0, num_of_samples - window_size + 1, step):
            end = start + window_size
            window = signal[start:end]
            all_windows.append(window)
            all_labels.append(class_id)

    windows_emg = np.stack(all_windows)
    y = np.array(all_labels)

    # print resultats attendus
    print("windows_emg:", windows_emg.shape)
    print("y:", y.shape)

    unique, counts = np.unique(y, return_counts=True)
    for uniq, count in zip(unique, counts):
        print(f"Classe {uniq}: {count} fenetres")

    return windows_emg, y


if __name__ == "__main__":
    dataset_builder()
