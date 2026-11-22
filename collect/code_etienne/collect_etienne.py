"""Collecte d'une session EMG (3 gestes) avec le GUI libemg.

    python collect/code_etienne/collect_etienne.py            # session de config.py
    python collect/code_etienne/collect_etienne.py --session 4
"""

import argparse
import atexit
import datetime
import shutil
import subprocess
import sys

import numpy as np

try:
    from config import Config
except ImportError:
    from code_etienne.config import Config

config = Config()
_ACTIVE_STREAMERS = []


def stop_streamer(streamer, timeout_s=5.0):
    """Arrete le processus du streamer libemg sans laisser sifibridge tenir le Bluetooth.

    On demande d'abord l'arret normal (libemg arrete l'acquisition, met le
    capteur en veille, se deconnecte et tue sifibridge). Cet arret n'a lieu
    qu'a l'arrivee du paquet suivant : si le capteur est eteint, il ne vient
    jamais, et on tue alors tout l'arbre de processus (streamer + sifibridge).
    """
    if streamer in _ACTIVE_STREAMERS:
        _ACTIVE_STREAMERS.remove(streamer)
    if streamer is None or not streamer.is_alive():
        return
    streamer.signal.set()
    streamer.join(timeout_s)
    if streamer.is_alive():
        print("Streamer bloque : arret force (streamer + sifibridge).")
        if sys.platform == "win32":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(streamer.pid)], capture_output=True)
        else:
            streamer.terminate()
        streamer.join(2.0)


def _stop_all_streamers():
    for streamer in list(_ACTIVE_STREAMERS):
        stop_streamer(streamer)


def sensor_in_use(tag="emg"):
    """True si un autre programme diffuse deja le capteur.

    libemg nomme sa memoire partagee 'emg' pour tout le monde : un 2e streamer
    la reutilise et les deux ecrivent dans le meme tampon (mesure : deux tests
    lances a 3 s d'ecart -> repos identique au bit pres, compteur remis a zero
    par l'autre, 3 fenetres fausses).
    """
    from multiprocessing import shared_memory
    try:
        existing = shared_memory.SharedMemory(name=tag, create=False)
    except FileNotFoundError:
        return False
    existing.close()
    return True


def _start_streamer(shared_memory_items, streaming):
    # Streamer libemg dont l'arret ne met pas le BioPoint en veille profonde
    # (sinon il faut appuyer sur son bouton avant chaque lancement).
    try:
        from sifi_streamer import start_biopoint_streamer
    except ImportError:
        from code_etienne.sifi_streamer import start_biopoint_streamer
    return start_biopoint_streamer(config.DEVICE_NAME, shared_memory_items, config.NOTCH, streaming=streaming,
                                   deep_sleep_on_exit=config.SLEEP_SENSOR_ON_EXIT)


def prepare_streamer(streaming=False, return_streamer=False):
    """Demarre le flux EMG du BioPoint et retourne l'OnlineDataHandler.

    Avec return_streamer=True, retourne (streamer, odh) : appeler
    stop_streamer(streamer) dans un finally. Sinon le streamer est arrete
    automatiquement a la sortie du programme.

    Le filtrage (passe-bande 20-450 Hz + coupe-bande secteur) est fait par le
    capteur lui-meme, a la collecte comme en temps reel. On n'ajoute donc
    aucun filtre logiciel : l'entrainement et le temps reel voient exactement
    le meme signal.
    """
    import libemg

    if sensor_in_use():
        raise RuntimeError("Le capteur est deja utilise par un autre programme (test, collecte ou pont de jeu). "
                           "Ferme-le d'abord : deux lecteurs en meme temps melangent leurs donnees.")
    shared_memory_items = [
        ["emg", (config.EMG_FS, 1), np.double],
        ["emg_count", (1, 1), np.int32],
    ]
    streamer, smi = _start_streamer(shared_memory_items, streaming)
    if not _ACTIVE_STREAMERS:
        # Enregistre APRES l'import de multiprocessing (par libemg) : atexit
        # est LIFO, donc cet arret propre passe avant que multiprocessing tue
        # les processus daemon et laisse sifibridge orphelin.
        atexit.unregister(_stop_all_streamers)
        atexit.register(_stop_all_streamers)
    _ACTIVE_STREAMERS.append(streamer)
    odh = libemg.data_handler.OnlineDataHandler(shared_memory_items=smi)
    return (streamer, odh) if return_streamer else odh


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
