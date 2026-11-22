"""Entrainement et evaluation du modele EMG a 3 gestes.

    python collect/code_etienne/trainning_etienne.py --sessions 3
    python collect/code_etienne/trainning_etienne.py --sessions 1,2 --train-reps 2,0,4,1,5 --test-reps 3,6
    python collect/code_etienne/trainning_etienne.py --sessions 1 --test-sessions 2   # test inter-session

Le score "temps reel" rejoue les repetitions de test dans le meme decodeur
que celui utilise en jeu (hysteresis + vote + suivi de derive).
"""

import argparse
import datetime

import numpy as np

try:
    from config import Config
    from emg_core import OnlineDecoder, ThreeGestureModel, extract_features, frame_signal, load_session, noise_profile
except ImportError:
    from code_etienne.config import Config
    from code_etienne.emg_core import (OnlineDecoder, ThreeGestureModel, extract_features, frame_signal,
                                       load_session, noise_profile)

config = Config()


def parse_int_list(raw_value):
    return [int(part.strip()) for part in str(raw_value).split(",") if part.strip()]


def load_recordings(sessions):
    recordings = []
    for session in sessions:
        session_recordings = load_session(config.session_dir(session), config.GESTURES)
        for recording in session_recordings:
            recording["session"] = session
        recordings.extend(session_recordings)
        print(f"S{session} : {len(session_recordings)} enregistrements charges ({', '.join(config.GESTURES)})")
    return recordings


def windows_of(recording):
    return frame_signal(recording["signal"], config.WINDOW_SIZE, config.WINDOW_INC)


def session_noise(recordings):
    """Profil du bruit de repos de chaque session presente dans `recordings`."""
    profiles = {}
    for session in sorted({rec["session"] for rec in recordings}):
        rest = [windows_of(rec) for rec in recordings if rec["session"] == session and rec["label"] == 0]
        if not rest:
            raise ValueError(f"S{session} : aucun enregistrement de repos.")
        profiles[session] = noise_profile(np.vstack(rest), config.N_LAGS)
    return profiles


def build_features(recordings, noise=None):
    """Features de toutes les fenetres. `noise` : profil unique, ou None pour
    utiliser le bruit de repos propre a chaque session (cas de l'entrainement)."""
    profiles = session_noise(recordings) if noise is None else None
    log_rms, shape, labels = [], [], []
    for recording in recordings:
        rec_noise = noise if profiles is None else profiles[recording["session"]]
        rec_rms, rec_shape = extract_features(windows_of(recording), config.N_LAGS, rec_noise)
        log_rms.append(rec_rms)
        shape.append(rec_shape)
        labels.append(np.full(len(rec_rms), recording["label"]))
    return np.concatenate(log_rms), np.vstack(shape), np.concatenate(labels)


def fit_model(recordings):
    model = ThreeGestureModel()
    model.gestures = list(config.GESTURES)
    model.fs = config.EMG_FS
    model.window_size = config.WINDOW_SIZE
    model.window_inc = config.WINDOW_INC
    model.n_lags = config.N_LAGS
    noise = np.mean(list(session_noise(recordings).values()), axis=0)
    model.fit(*build_features(recordings), noise=noise)
    return model


def replay_accuracy(model, recordings, skip_s=0.4):
    """Rejoue les enregistrements dans le decodeur temps reel.

    Les `skip_s` premieres secondes de chaque fichier ne sont pas notees :
    c'est le temps de reaction du vote quand on passe d'un geste a l'autre.
    """
    decoder = OnlineDecoder(model, vote_window=config.VOTE_WINDOW, adapt=config.ADAPT)
    ordered = sorted(recordings, key=lambda rec: (rec["session"], rec["rep"], rec["label"]))
    confusion = np.zeros((3, 3), dtype=int)
    for recording in ordered:
        windows = frame_signal(recording["signal"], config.WINDOW_SIZE, config.WINDOW_INC)
        for index, window in enumerate(windows):
            output = decoder.update(window)
            if index * config.WINDOW_INC / config.EMG_FS >= skip_s:
                confusion[recording["label"], output["label"]] += 1
    return confusion


def print_confusion(confusion):
    width = max(len(name) for name in config.GESTURES) + 2
    print(" " * width + "".join(f"{name:>{width}}" for name in config.GESTURES) + "   <- predit")
    for row, name in enumerate(config.GESTURES):
        total = max(confusion[row].sum(), 1)
        print(f"{name:<{width}}" + "".join(f"{confusion[row, col] / total:>{width}.1%}" for col in range(3)))


def evaluate(model, test_recordings):
    log_rms, shape, labels = build_features(test_recordings, noise=model.noise)
    window_accuracy = float(np.mean(model.predict(log_rms, shape) == labels))
    confusion = replay_accuracy(model, test_recordings)
    replay = float(np.trace(confusion) / max(confusion.sum(), 1))
    print(f"Exactitude par fenetre (sans memoire) : {window_accuracy:.1%}")
    print(f"Exactitude temps reel simule          : {replay:.1%}")
    print_confusion(confusion)
    return window_accuracy, replay


def save_model(model, sessions):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = "-".join(str(session) for session in sessions)
    path = config.MODEL_PATH / f"{config.MODEL_PREFIX}_S{tag}_{timestamp}.json"
    model.save(path)
    print(f"Modele sauvegarde : {path}")
    return path


def train_and_test(sessions=None, train_reps=None, test_reps=None, test_sessions=None, save=True):
    sessions = list(config.SESSIONS_TRAIN) if sessions is None else list(sessions)
    train_reps = list(config.TRAIN_REPS) if train_reps is None else list(train_reps)
    test_reps = list(config.TEST_REPS) if test_reps is None else list(test_reps)
    overlap = sorted(set(train_reps) & set(test_reps))
    if overlap and not test_sessions:
        raise ValueError(f"Les repetitions {overlap} sont a la fois dans train et test.")

    recordings = load_recordings(sessions)
    info = {"sessions": sessions, "trained": datetime.datetime.now().isoformat(timespec="seconds")}

    if test_sessions:
        # Test inter-session : toutes les repetitions des sessions de train,
        # evaluation sur des sessions jamais vues.
        print(f"\n== Entrainement sur S{sessions}, test sur S{list(test_sessions)}")
        model = fit_model(recordings)
        info["window_accuracy"], info["replay_accuracy"] = evaluate(model, load_recordings(test_sessions))
    else:
        train = [rec for rec in recordings if rec["rep"] in train_reps]
        test = [rec for rec in recordings if rec["rep"] in test_reps]
        if not train or not test:
            raise ValueError("Split vide : verifie --train-reps et --test-reps.")
        print(f"\n== Evaluation : train reps {train_reps}, test reps {test_reps}")
        info["window_accuracy"], info["replay_accuracy"] = evaluate(fit_model(train), test)
        # Le modele final utilise toutes les repetitions : plus de donnees,
        # le score ci-dessus reste une estimation honnete (prudente).
        model = fit_model(recordings)

    model.info.update(info)
    rest_uv, active_uv = np.exp(model.rest_level) * 1e6, np.exp(model.active_level) * 1e6
    print(f"\nNiveau repos {rest_uv:.1f} uV | niveau actif {active_uv:.1f} uV | rapport {active_uv / rest_uv:.1f}x")
    if info["replay_accuracy"] < 0.90:
        print("ATTENTION : moins de 90 %. Verifie la position du capteur et refais une collecte avant de jouer.")
    if save:
        save_model(model, sessions)
    return model, info


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entraine et evalue le modele EMG a 3 gestes.")
    parser.add_argument("--sessions", "--session", type=str, default=None,
                        help="Sessions d'entrainement separees par des virgules, ex. 3 ou 1,2")
    parser.add_argument("--train-reps", type=str, default=None, help="Repetitions d'entrainement, ex. 2,0,4,1,5")
    parser.add_argument("--test-reps", type=str, default=None, help="Repetitions de test, ex. 3,6")
    parser.add_argument("--test-sessions", type=str, default=None,
                        help="Si fourni : entraine sur --sessions (toutes les reps) et teste sur ces sessions")
    parser.add_argument("--no-save", action="store_true", help="Ne pas sauvegarder le modele")
    args = parser.parse_args()
    train_and_test(
        sessions=parse_int_list(args.sessions) if args.sessions else None,
        train_reps=parse_int_list(args.train_reps) if args.train_reps else None,
        test_reps=parse_int_list(args.test_reps) if args.test_reps else None,
        test_sessions=parse_int_list(args.test_sessions) if args.test_sessions else None,
        save=not args.no_save,
    )
