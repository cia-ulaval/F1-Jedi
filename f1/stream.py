from typing import List, Dict, Tuple
import os
import threading
import socket

import numpy as np

from libemg import (
    streamers, data_handler, filtering, gui,
    emg_predictor, feature_extractor
)

# ========= CONFIG GLOBALE ========= #

WINDOW_SIZE_MS = 200      # ms
WINDOW_INC_MS  = 20       # ms

FS_EMG = 2000
FS_IMU = 100
FS_PPG = 100

WINDOW_SIZE_EMG = int(FS_EMG * WINDOW_SIZE_MS / 1000)  # 2000*0.2=400
WINDOW_INC_EMG  = int(FS_EMG * WINDOW_INC_MS  / 1000)  # 2000*0.02=40

WINDOW_SIZE_IMU = int(FS_IMU * WINDOW_SIZE_MS / 1000)  # 100*0.2=20
WINDOW_INC_IMU  = int(FS_IMU * WINDOW_INC_MS  / 1000)  # 100*0.02=2

WINDOW_SIZE_PPG = int(FS_PPG * WINDOW_SIZE_MS / 1000)
WINDOW_INC_PPG  = int(FS_PPG * WINDOW_INC_MS  / 1000)

CLASSES = [0, 1, 2, 3, 4]      # 0 = repos, 1..4 = gestes
REPS    = [0, 1, 2, 3, 4, 5]
SUBJECTS = list(range(8))

DATA_ROOT = "data"   # new_data/S0, new_data/S1, ...

MODEL_PATH = "models/multimodal_model"
SCALER_PATH = "models/multimodal_scaler.npy"


# ========= 1. COLLECTE DE DONNÉES ========= #

class CustomGui(gui.GUI):
    def __init__(self,
                 online_data_handler,
                 args,
                 width=1920,
                 height=1080,
                 debug=False,
                 gesture_width=500,
                 gesture_height=500,
                 clean_up_on_kill=False):
        super().__init__(
            online_data_handler=online_data_handler,
            args=args,
            width=width,
            height=height,
            debug=debug,
            gesture_width=gesture_width,
            gesture_height=gesture_height,
            clean_up_on_kill=clean_up_on_kill
        )

    def download_gestures(self, gesture_ids, folder, download_imgs=True,
                          download_gifs=False, redownload=False):
        import requests
        from pathlib import Path

        git_url = "https://raw.githubusercontent.com/cia-ulaval/F1-team-1/main/f1/"
        gesture_data = {
            "1": "No_Motion",
            "2": "Wrist_Flexion",
            "3": "Hand_Closed",
            "4": "Hand_Opened",
            "5": "Wrist_Extension",
        }

        folder_path = Path(folder)
        folder_path.mkdir(parents=True, exist_ok=True)

        for gid in gesture_ids:
            idx = str(gid)
            if idx not in gesture_data:
                print(f"[WARN] Gesture ID {gid} unknown")
                continue

            img_file = gesture_data[idx] + ".jpg"
            img_path = folder_path / img_file
            img_url  = git_url + "images/" + img_file

            if download_imgs and (not img_path.exists() or redownload):
                print(f"Downloading {img_file}")
                r = requests.get(img_url)
                if r.status_code == 200:
                    with open(img_path, "wb") as f:
                        f.write(r.content)
                else:
                    print(f"Failed to download {img_file} : {r.status_code}")


def collect_data_for_subject(subject_id: int):
    streamer, smm = streamers.sifi_biopoint_streamer(
        name='BioPoint_v1_3',
        ecg=False, imu=True, ppg=True, eda=False, emg=True,
        filtering=True, emg_notch_freq=60
    )
    odh = data_handler.OnlineDataHandler(smm)

    args = {
        "media_folder": "images/",
        "data_folder": os.path.join(DATA_ROOT, f"S{subject_id}/"),
        "num_reps": 6,
        "rep_time": 5,
        "rest_time": 3,
        "auto_advance": True
    }

    gui_ = CustomGui(odh, args=args, debug=False)
    gui_.download_gestures([1,2,3,4,5], "images/")
    gui_.start_gui()


# ========= 2. OFFLINE : CHARGEMENT & PRÉTRAITEMENT ========= #

def build_regex_filters_emg():
    return [
        data_handler.RegexFilter(
            left_bound="C_", right_bound="_R_",
            values=[str(i) for i in REPS],
            description='reps'
        ),
        data_handler.RegexFilter(
            left_bound="_R_", right_bound="_emg.csv",
            values=[str(i) for i in CLASSES],
            description='classes'
        )
    ]


def build_regex_filters_imu():
    # TODO : adapter le suffixe selon comment libemg sauvegarde l’IMU
    return [
        data_handler.RegexFilter(
            left_bound="C_", right_bound="_R_",
            values=[str(i) for i in REPS],
            description='reps'
        ),
        data_handler.RegexFilter(
            left_bound="_R_", right_bound="_imu.csv",
            values=[str(i) for i in CLASSES],
            description='classes'
        )
    ]


def build_regex_filters_ppg():
    # TODO : adapter le suffixe selon les fichiers PPG générés
    return [
        data_handler.RegexFilter(
            left_bound="C_", right_bound="_R_",
            values=[str(i) for i in REPS],
            description='reps'
        ),
        data_handler.RegexFilter(
            left_bound="_R_", right_bound="_ppg.csv",
            values=[str(i) for i in CLASSES],
            description='classes'
        )
    ]


def get_emg_windows(folder: str) -> Tuple[np.ndarray, Dict]:
    odh = data_handler.OfflineDataHandler()
    odh.get_data(folder_location=folder, regex_filters=build_regex_filters_emg())
    windows, metadata = odh.parse_windows(WINDOW_SIZE_EMG, WINDOW_INC_EMG)

    fi = filtering.Filter(FS_EMG)
    fi.install_filters({"name": "standardize", "data": odh})
    fi.install_filters({"name": "notch", "cutoff": 60, "bandwidth": 3})
    fi.install_filters({"name": "bandpass", "cutoff": [20, 450], "order": 4})

    windows_filt = fi.filter(windows)
    return windows_filt, metadata


def get_imu_windows(folder: str) -> Tuple[np.ndarray, Dict]:
    """
    On suppose que libemg peut charger l’IMU de la même façon
    que l’EMG, avec un autre suffixe (_imu.csv).
    """
    odh = data_handler.OfflineDataHandler()
    odh.get_data(folder_location=folder, regex_filters=build_regex_filters_imu())
    windows, metadata = odh.parse_windows(WINDOW_SIZE_IMU, WINDOW_INC_IMU)

    # Standardisation simple
    fi = filtering.Filter(FS_IMU)
    fi.install_filters({"name": "standardize", "data": odh})
    windows_filt = fi.filter(windows)

    # Dérivée pour limiter effet orientation
    # windows_filt shape: (n_windows, window_len, n_channels)
    windows_diff = np.diff(windows_filt, axis=1, prepend=windows_filt[:, :1, :])
    return windows_diff, metadata


def get_ppg_windows(folder: str) -> Tuple[np.ndarray, Dict]:
    odh = data_handler.OfflineDataHandler()
    odh.get_data(folder_location=folder, regex_filters=build_regex_filters_ppg())
    windows, metadata = odh.parse_windows(WINDOW_SIZE_PPG, WINDOW_INC_PPG)

    fi = filtering.Filter(FS_PPG)
    fi.install_filters({"name": "standardize", "data": odh})
    # TODO : si libemg supporte un band-stop, l’ajouter ici (0.66–3 Hz)
    windows_filt = fi.filter(windows)

    # TODO : PCA sur canaux R+IR si tu as les 4 canaux (Blue, Green, Red, IR)
    # pour l’instant, on garde tel quel
    return windows_filt, metadata


# ========= 3. FEATURES MULTIMODALES ========= #

def extract_emg_features(windows_emg: np.ndarray) -> Dict[str, np.ndarray]:
    fe = feature_extractor.FeatureExtractor()
    feat_dict = fe.extract_feature_group("LS4", windows_emg)  # LS, MFL, MSR, WAMP
    return feat_dict


def extract_simple_stats(windows: np.ndarray, prefix: str) -> Dict[str, np.ndarray]:
    """
    Features simples tempo/énergie pour IMU/PPG.
    windows: (n_windows, window_len, n_channels)
    """
    mean = windows.mean(axis=1)
    std  = windows.std(axis=1)
    energy = (windows ** 2).sum(axis=1)
    wl = np.sum(np.abs(np.diff(windows, axis=1)), axis=1)

    return {
        f"{prefix}_MEAN": mean,
        f"{prefix}_STD": std,
        f"{prefix}_ENERGY": energy,
        f"{prefix}_WL": wl,
    }


def concat_feature_dicts(dicts: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for d in dicts:
        for k, v in d.items():
            if k not in out:
                out[k] = v
            else:
                out[k] = np.concatenate([out[k], v], axis=0)
    return out


# ========= 4. STANDARDISATION GLOBALE ========= #

def standardize_feature_dict_global(feature_dict: Dict[str, np.ndarray]) -> Tuple[Dict[str, np.ndarray], Dict[str, Tuple[np.ndarray, np.ndarray]]]:
    """
    Standardisation z-score par feature (global sur tous les sujets).
    Retourne un dict standardisé + un dict {feature_name: (mean, std)}.
    """
    scaled = {}
    scaler = {}
    for k, arr in feature_dict.items():
        mean = arr.mean(axis=0)
        std  = arr.std(axis=0) + 1e-8
        scaled[k] = (arr - mean) / std
        scaler[k] = (mean, std)
    return scaled, scaler


def apply_scaler(feature_dict: Dict[str, np.ndarray],
                 scaler: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> Dict[str, np.ndarray]:
    out = {}
    for k, arr in feature_dict.items():
        mean, std = scaler[k]
        out[k] = (arr - mean) / std
    return out


# ========= 5. ENTRAÎNEMENT OFFLINE ========= #

def prepare_multimodal_model():
    all_feat_dicts = []
    all_labels = []

    for sid in SUBJECTS:
        folder = os.path.join(DATA_ROOT, f"S{sid}")
        print(f"[INFO] Processing subject S{sid}...")

        # EMG
        emg_windows, meta_emg = get_emg_windows(folder)
        emg_feat = extract_emg_features(emg_windows)

        # IMU
        imu_windows, meta_imu = get_imu_windows(folder)
        imu_feat = extract_simple_stats(imu_windows, prefix="IMU")

        # PPG
        ppg_windows, meta_ppg = get_ppg_windows(folder)
        ppg_feat = extract_simple_stats(ppg_windows, prefix="PPG")

        # Vérification cohérence labels
        labels_emg = meta_emg['classes']
        labels_imu = meta_imu['classes']
        labels_ppg = meta_ppg['classes']

        assert np.array_equal(labels_emg, labels_imu), "Labels EMG/IMU mismatch"
        assert np.array_equal(labels_emg, labels_ppg), "Labels EMG/PPG mismatch"

        feat_subject = concat_feature_dicts([emg_feat, imu_feat, ppg_feat])
        all_feat_dicts.append(feat_subject)
        all_labels.append(labels_emg)

    # concat sur tous les sujets
    merged_feats: Dict[str, np.ndarray] = {}
    for fd in all_feat_dicts:
        for k, v in fd.items():
            if k not in merged_feats:
                merged_feats[k] = v
            else:
                merged_feats[k] = np.concatenate([merged_feats[k], v], axis=0)

    merged_labels = np.concatenate(all_labels, axis=0)

    # standardisation globale
    scaled_feats, scaler = standardize_feature_dict_global(merged_feats)

    feature_dic_for_libemg = {
        "training_features": scaled_feats,
        "training_labels": merged_labels
    }

    # ==== Modèle : LDA ou RF ==== #
    model = emg_predictor.EMGClassifier("LDA", {})  # ou "RF", {...}
    model.fit(feature_dictionary=feature_dic_for_libemg)
    model.add_velocity(None, merged_labels)  # option : tu peux passer les windows EMG
    model.add_rejection(0.5)

    # Sauvegarde LibEMG
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    model.save(MODEL_PATH)

    # Sauvegarde scaler
    np.save(SCALER_PATH, scaler, allow_pickle=True)

    print("[INFO] Multimodal model + scaler saved.")


# ========= 6. ONLINE : STREAM + PREDICTION + CONTRÔLE ========= #

def run_online_multimodal_control():
    from drive import accelerate, brake, steer_left, steer_right, reset_controls

    # Charger modèle
    model = emg_predictor.EMGClassifier.from_file(MODEL_PATH)
    scaler = np.load(SCALER_PATH, allow_pickle=True).item()

    streamer, smm = streamers.sifi_biopoint_streamer(
        name='BioPoint_v1_3',
        ecg=False, imu=True, ppg=True, eda=False, emg=True,
        filtering=True, emg_notch_freq=60
    )
    odh = data_handler.OnlineDataHandler(smm)
    odh.analyze_hardware(60)

    # Filtre EMG online
    fi_emg_online = filtering.Filter(FS_EMG)
    fi_emg_online.install_filters({"name": "standardize", "data": odh})
    fi_emg_online.install_filters({"name": "notch", "cutoff": 60, "bandwidth": 3})
    fi_emg_online.install_filters({"name": "bandpass", "cutoff": [20, 450], "order": 4})
    odh.install_filter(fi_emg_online)

    # On va utiliser OnlineEMGClassifier juste comme boucle temps réel,
    # mais on veut nos propres features multimodales → on fait une boucle perso.

    UDP_IP = "127.0.0.1"
    UDP_PORT = 12346
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_IP, UDP_PORT))

    print(f"[INFO] Listening on {UDP_IP}:{UDP_PORT}...")

    # Exemple minimal : suppose que ton process libemg envoie "class velocity"
    while True:
        data, _ = sock.recvfrom(1024)
        msg = data.decode("utf-8").strip()
        try:
            pred_str, vel_str = msg.split(" ")
            pred = int(pred_str)
            vel  = float(vel_str)

            # Mapping classes → contrôle
            if pred == 0:  # repos
                reset_controls()
            elif pred == 1:  # flexion pouce
                accelerate()
            elif pred == 2:  # extension / frein
                brake()
            elif pred == 3:  # rotation droite
                steer_right()
            elif pred == 4:  # rotation gauche
                steer_left()

        except Exception as e:
            print(f"[ERROR] parsing message '{msg}': {e}")


# ========= 7. ENTRY POINT ========= #

if __name__ == "__main__":
    STAGE = 1  # 0: collect, 1: train, 2: online

    if STAGE == 0:
        collect_data_for_subject(subject_id=0)   # change l’id

    elif STAGE == 1:
        prepare_multimodal_model()

    elif STAGE == 2:
        run_online_multimodal_control()
