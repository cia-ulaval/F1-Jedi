"""Collecte d'une session EMG (3 gestes) avec le GUI libemg.

    python collect/code_etienne/collect_etienne.py            # session de config.py
    python collect/code_etienne/collect_etienne.py --session 4
"""

import argparse
import datetime
import shutil
import sys

import numpy as np

try:
    from config import Config
except ImportError:
    from code_etienne.config import Config

config = Config()


def prepare_streamer(streaming=False):
    """Demarre le flux EMG du BioPoint et retourne l'OnlineDataHandler.

    Le filtrage (passe-bande 20-450 Hz + coupe-bande secteur) est fait par le
    capteur lui-meme, a la collecte comme en temps reel. On n'ajoute donc
    aucun filtre logiciel : l'entrainement et le temps reel voient exactement
    le meme signal.
    """
    import libemg

    shared_memory_items = [
        ["emg", (config.EMG_FS, 1), np.double],
        ["emg_count", (1, 1), np.int32],
    ]
    _streamer, smi = libemg.streamers.sifi_biopoint_streamer(
        name=config.DEVICE_NAME,
        shared_memory_items=shared_memory_items,
        ecg=False,
        emg=True,
        eda=False,
        imu=False,
        ppg=False,
        filtering=True,
        emg_notch_freq=config.NOTCH,
        streaming=streaming,
    )
    return libemg.data_handler.OnlineDataHandler(shared_memory_items=smi)


def prepare_media_folder(gui=None):
    """Construit le dossier d'images qui ne contient QUE les 3 gestes.

    Le GUI libemg enregistre une classe par image presente dans le dossier,
    numerotee par ordre alphabetique. On regenere donc le dossier a chaque
    collecte a partir de config.GESTURES.
    """
    source, target = config.MEDIA_SOURCE, config.MEDIA_FOLDER
    missing = [name for name in config.GESTURES if not (source / f"{name}.png").exists()]
    if missing and gui is not None:
        gui.download_gestures([config.GESTURE_IDS[name] for name in missing], str(source) + "/")
    missing = [name for name in config.GESTURES if not (source / f"{name}.png").exists()]
    if missing:
        raise FileNotFoundError(f"Images manquantes dans {source} : {missing}")

    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    for name in config.GESTURES:
        shutil.copy(source / f"{name}.png", target / f"{name}.png")
    return target


def ensure_folder(session):
    """Prepare dataset/S<session>. Si des donnees existent deja, demande quoi faire."""
    dataset_path = config.session_dir(session)
    dataset_path.mkdir(parents=True, exist_ok=True)
    if not any(dataset_path.iterdir()):
        return dataset_path

    print(f"Des donnees existent deja dans '{dataset_path}'.")
    while True:
        choice = input("Action - [E]craser, [A]rchiver l'ancienne et recommencer, [Q]uitter : ").strip().lower()
        if choice in ("e", "ecraser", "o", "overwrite"):
            confirm = input(f"Supprimer TOUT le contenu de '{dataset_path}' ? Tape 'oui' pour confirmer : ").strip().lower()
            if confirm in ("oui", "yes"):
                shutil.rmtree(dataset_path)
                dataset_path.mkdir(parents=True)
                return dataset_path
            print("Suppression annulee.")
        elif choice in ("a", "archiver", "r", "rename"):
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            backup = dataset_path.with_name(f"{dataset_path.name}_archive_{timestamp}")
            dataset_path.rename(backup)
            dataset_path.mkdir(parents=True)
            print(f"Ancienne session deplacee vers '{backup}'.")
            return dataset_path
        elif choice in ("q", "quitter", "c", "cancel"):
            print("Collecte annulee.")
            sys.exit(0)
        else:
            print("Choix invalide : E, A ou Q.")


def collect_data(session=None):
    import libemg

    session = config.SESSION_COLLECT if session is None else session
    dataset_path = ensure_folder(session)
    print(f"Collecte de la session S{session} -> {dataset_path}")
    print(f"Gestes : {', '.join(config.GESTURES)} | {config.NUM_REPS} repetitions de {config.REP_TIME} s")

    odh = prepare_streamer()
    args = {
        "online_data_handler": odh,
        "media_folder": str(config.MEDIA_FOLDER) + "/",
        "data_folder": str(dataset_path) + "/",
        "num_reps": config.NUM_REPS,
        "rep_time": config.REP_TIME,
        "rest_time": config.REST_TIME,
        "auto_advance": True,
    }
    gui = libemg.gui.GUI(odh, args=args, debug=False, width=config.GUI_WIDTH, height=config.GUI_HEIGHT)
    prepare_media_folder(gui)
    gui.start_gui()

    print(f"Collecte terminee : {dataset_path}")
    print(f"Etape suivante : python collect/code_etienne/trainning_etienne.py --sessions {session}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collecte d'une session EMG a 3 gestes.")
    parser.add_argument("--session", type=int, default=config.SESSION_COLLECT, help="Numero de session, ex. 3 pour dataset/S3")
    collect_data(parser.parse_args().session)
