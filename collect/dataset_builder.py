import numpy as np
import pandas as pd
import os
from pathlib import Path
from validate_dataset import validate_subject_folder

DATA_DIR = Path("./data")

CLASS_ID_MAP = {
    "Hand_Open": 0,
    "Hand_Close": 1,
    "No_Motion": 2,
    "Wrist_Extension": 3,
    "Wrist_Flexion": 4
}


def get_class_id_from_file(file_path, class_map=CLASS_ID_MAP):
    pose_name = file_path.stem
    return class_map.get(pose_name)


def create_windows(signal, window_size, step):

    windows = []
    num_samples = signal.shape[0]

    # si le signal est plus court que la taille de la fenêtre, on ignore le fichier
    if num_samples < window_size:
        return windows

    # fenetrage
    for start in range(0, num_samples - window_size + 1, step):
        end = start + window_size
        window = signal[start:end]
        windows.append(window)

    return windows


def process_file(file_path, window_size, step, class_map=CLASS_ID_MAP):

    # Identifier la classe correspondant au fichier
    class_id = get_class_id_from_file(file_path, class_map)

    # Si le fichier ne correspond à aucune classe connue
    if class_id is None:
        print(f"Ignoré (classe inconnue): {file_path.name}")
        return [], []

    # charger le csv
    data_frame = pd.read_csv(file_path)
    signal = data_frame.values

    # fenetrage
    windows = create_windows(signal, window_size, step)

    # creer un label pour chaque fenetre
    labels = [class_id] * len(windows)

    return windows, labels


def dataset_builder(fs_emg=2000, window_size_ms=200, step_ms=20):
    data_dir = DATA_DIR

    window_size = int(window_size_ms * fs_emg / 1000)  # taille fenetre = 400
    step = int(step_ms * fs_emg / 1000)  # taille step = 2

    all_windows = []
    all_labels = []

    # trouver les fichiers csv
    csv_files = sorted(data_dir.rglob("*.csv"))

    for file_path in csv_files:
        file_windows, file_labels = process_file(file_path, window_size, step)

        all_windows.extend(file_windows)
        all_labels.extend(file_labels)

    # conversion en numpy arrays
    windows_emg = np.stack(all_windows)
    # conversion windows_emg en 2D
    windows_emg = windows_emg.reshape(len(windows_emg), -1)

    y = np.array(all_labels)

    # print resultats attendus
    print("windows_emg:", windows_emg.shape)
    print("y:", y.shape)

    unique, counts = np.unique(y, return_counts=True)
    for uniq, count in zip(unique, counts):
        print(f"Classe {uniq}: {count} fenetres")

    return windows_emg, y


def save_dataset_to_csv(windows_emg, y, output_file="./dataset/emg_dataset.csv"):
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    df_X = pd.DataFrame(windows_emg)
    df_X["label"] = y

    df_X.to_csv(output_file, index=False)

    print("Dataset sauvegardé dans :", output_file)


def main():
    folder = "./data"
    if not os.path.exists(folder):
        print(f"ERROR: {folder} does not exist")
        exit(1)
    else:
        for subject in os.listdir(folder):
            validate_subject_folder(os.path.join(folder, subject))

    print()
    windows_emg, labels = dataset_builder()
    save_dataset_to_csv(windows_emg, labels)


if __name__ == "__main__":
    main()
