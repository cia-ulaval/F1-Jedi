"""Test temps reel du modele EMG a 3 gestes (affichage console).

    python collect/code_etienne/realtime_test_etienne.py
    python collect/code_etienne/realtime_test_etienne.py --model_path models/f1jedi_3g_S3_....json
    python collect/code_etienne/realtime_test_etienne.py --replay 2     # sans capteur : rejoue dataset/S2

Ce module fournit aussi les briques reutilisees par realtime_game_bridge.py.
"""

import argparse
import time
from pathlib import Path

import numpy as np

try:
    from config import Config
    from emg_core import OnlineDecoder, ThreeGestureModel, load_session
except ImportError:
    from code_etienne.config import Config
    from code_etienne.emg_core import OnlineDecoder, ThreeGestureModel, load_session

config = Config()
MODEL_PATH = None


# ------------------------------------------------------------------ modele --
def find_last_model(model_dir=None):
    model_dir = Path(config.MODEL_PATH if model_dir is None else model_dir)
    model_files = list(model_dir.glob(f"{config.MODEL_PREFIX}_*.json"))
    if not model_files:
        raise FileNotFoundError(
            f"Aucun modele {config.MODEL_PREFIX}_*.json dans {model_dir}. "
            "Lance d'abord trainning_etienne.py."
        )
    return max(model_files, key=lambda file: file.stat().st_mtime)


def load_model(model_path=None):
    path = find_last_model() if not model_path else Path(model_path)
    if not path.is_absolute() and not path.exists():
        path = config.REPO_ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"Modele introuvable : {path}")
    model = ThreeGestureModel.load(path)
    if model.fs != config.EMG_FS:
        raise ValueError(f"Le modele a ete entraine a {model.fs} Hz, la config est a {config.EMG_FS} Hz.")
    print(f"Modele charge : {path}")
    print(f"Gestes : {', '.join(model.gestures)}")
    return model


# ------------------------------------------------------- lecture du capteur --
class WindowReader:
    """Fournit la fenetre EMG la plus recente a chaque nouvel increment."""

    def __init__(self, odh, window_size, window_inc):
        self.odh = odh
        self.window_size = window_size
        self.window_inc = window_inc
        self._next_count = window_size

    def _read(self, num_samples):
        values, count = self.odh.get_data(N=num_samples)
        total = int(np.asarray(count["emg"]).reshape(-1)[0])
        return values["emg"], total

    def wait_for_emg(self, timeout_s):
        deadline = time.perf_counter() + timeout_s
        while time.perf_counter() < deadline:
            if self._read(1)[1] > 0:
                return True
            time.sleep(0.05)
        return False

    def reset(self):
        self.odh.reset()
        self._next_count = self.window_size

    def poll(self):
        """Retourne une fenetre 1-D (ordre chronologique) ou None s'il n'y a rien de neuf.

        Les fenetres avancent de window_inc echantillons exactement, comme a
        l'entrainement, meme si les donnees arrivent par paquets.
        """
        max_lag = 4 * self.window_inc
        data, total = self._read(self.window_size + max_lag)
        if total < self._next_count:
            return None
        if total - self._next_count > max_lag:
            # Trop de retard : on saute a la fenetre la plus recente au lieu
            # d'accumuler de la latence.
            self._next_count = total
        newer = total - self._next_count      # echantillons arrives apres la fin de la fenetre
        self._next_count += self.window_inc
        # Le tampon partage est range du plus recent au plus ancien.
        return np.asarray(data[newer:newer + self.window_size, 0], dtype=float)[::-1]

    def collect(self, duration_s, on_tick=None):
        """Accumule les fenetres recues pendant `duration_s` secondes."""
        windows = []
        deadline = time.perf_counter() + duration_s
        while time.perf_counter() < deadline:
            window = self.poll()
            if window is not None:
                windows.append(window)
            if on_tick is not None:
                on_tick()
            time.sleep(config.DELAY_BETWEEN_PREDICTIONS)
        return np.array(windows) if windows else np.empty((0, self.window_size))


class ReplayReader:
    """Meme interface que WindowReader, mais rejoue une session du dataset.

    Sert a essayer la boucle temps reel sans capteur. Les enregistrements
    sont joues repetition par repetition, en temps reel (ou acceleres).
    """

    def __init__(self, session, gestures, window_size, window_inc, fs, speed=1.0):
        recordings = load_session(config.session_dir(session), gestures)
        recordings.sort(key=lambda rec: (rec["rep"], rec["label"]))
        self.signal = np.concatenate([rec["signal"] for rec in recordings])
        self.truth = np.concatenate([np.full(len(rec["signal"]), rec["label"]) for rec in recordings])
        self.window_size, self.window_inc, self.fs, self.speed = window_size, window_inc, fs, speed
        self.reset()

    def wait_for_emg(self, timeout_s):
        return True

    def reset(self):
        self._start = time.perf_counter()
        self._next_count = self.window_size
        self._last_end = self.window_size

    def _position(self):
        return int((time.perf_counter() - self._start) * self.fs * self.speed)

    @property
    def finished(self):
        return self._next_count > len(self.signal)

    def current_truth(self):
        """Verite terrain a la fin de la derniere fenetre rendue."""
        return int(self.truth[min(self._last_end, len(self.truth)) - 1])

    def poll(self):
        # Toutes les fenetres sont rendues dans l'ordre, pas de window_inc
        # exact, meme en replay accelere (sinon le decodeur en saute).
        position = min(self._position(), len(self.signal))
        if position < self._next_count:
            return None
        end = self._next_count
        self._next_count += self.window_inc
        self._last_end = end
        return self.signal[end - self.window_size:end]

    collect = WindowReader.collect


def open_sensor():
    """Connecte le BioPoint et attend l'arrivee des premiers echantillons."""
    try:
        from collect_etienne import prepare_streamer
    except ImportError:
        from code_etienne.collect_etienne import prepare_streamer

    reader = WindowReader(prepare_streamer(streaming=False), config.WINDOW_SIZE, config.WINDOW_INC)
    print(f"Attente du capteur (max {config.TIMEOUT:.0f} s)...")
    if not reader.wait_for_emg(config.TIMEOUT):
        raise RuntimeError(f"Aucun echantillon EMG recu apres {config.TIMEOUT:.0f} s. Capteur allume et appaire ?")
    reader.reset()
    return reader


# ------------------------------------------------------------- calibration --
def calibrate_rest(reader, decoder, duration_s=None, on_tick=None):
    """Mesure le niveau de repos du jour. A faire bras detendu."""
    duration_s = config.REST_CALIBRATION_S if duration_s is None else duration_s
    print(f"Calibration du repos : detends la main pendant {duration_s:.0f} s...")
    # Les 0.5 premieres secondes laissent le temps de se detendre.
    reader.collect(0.5, on_tick)
    windows = reader.collect(duration_s, on_tick)
    if len(windows) == 0:
        print("  aucune donnee recue : niveau de repos du modele conserve.")
        return
    learned = float(np.exp(decoder.model.rest_level) * 1e6)
    measured = float(np.exp(decoder.calibrate_rest(windows)) * 1e6)
    print(f"  repos mesure {measured:.1f} uV (a l'entrainement : {learned:.1f} uV)")
    if measured > 3.0 * learned:
        print("  ATTENTION : repos bien plus fort qu'a l'entrainement (main pas detendue ou mauvais contact).")
        print(f"  Niveau plafonne a {3.0 * learned:.1f} uV. Replace le capteur ou recalibre avec R.")


def guided_recalibration(reader, decoder, duration_s=None, on_tick=None):
    """Recalibration rapide (~15 s) quand le decodage se degrade en cours de jeu.

    Retourne True si la recalibration a ete appliquee.
    """
    duration_s = config.RECALIBRATION_S if duration_s is None else duration_s
    captured = []
    print("\n=== Recalibration rapide ===")
    for name in decoder.model.gestures:
        print(f"  -> prepare : {name}")
        reader.collect(1.5, on_tick)
        print(f"     TIENS {name} pendant {duration_s:.0f} s")
        # On jette le debut : le geste n'est pas encore installe.
        reader.collect(0.5, on_tick)
        captured.append(reader.collect(duration_s, on_tick))
    print("  -> relache")
    try:
        levels = decoder.recalibrate(*captured)
    except ValueError as error:
        print(f"  {error}")
        print("  Ancien reglage conserve.\n")
        return False
    print(f"  OK : repos {levels['rest_uV']:.1f} uV, actif {levels['active_uV']:.1f} uV\n")
    return True


# -------------------------------------------------------------------- main --
def run_realtime_test(model_path=None, vote_window=None, adapt=None, replay=None, speed=1.0):
    model = load_model(model_path)
    vote_window = config.VOTE_WINDOW if vote_window is None else vote_window
    adapt = config.ADAPT if adapt is None else adapt
    decoder = OnlineDecoder(model, vote_window=vote_window, adapt=adapt)

    if replay is None:
        reader = open_sensor()
        calibrate_rest(reader, decoder)
    else:
        reader = ReplayReader(replay, model.gestures, model.window_size, model.window_inc, model.fs, speed)
        print(f"Mode replay : dataset/S{replay} ({len(reader.signal) / model.fs:.0f} s de signal, vitesse x{speed:g})")

    print("Prediction en temps reel. Ctrl+C pour arreter.")
    last_print, last_label, correct, total = 0.0, None, 0, 0
    try:
        while replay is None or not reader.finished:
            window = reader.poll()
            if window is None:
                time.sleep(config.DELAY_BETWEEN_PREDICTIONS)
                continue
            output = decoder.update(window)
            if replay is not None:
                total += 1
                correct += int(output["label"] == reader.current_truth())
            now = time.perf_counter()
            # Affiche a chaque changement de geste, sinon 4 fois par seconde.
            if output["label"] != last_label or now - last_print > 0.25:
                print(
                    f"{output['name']:<16} rms={output['rms_uV']:6.1f} uV  "
                    f"seuil={output['threshold_uV']:5.1f} uV  repos={output['rest_uV']:5.1f} uV"
                )
                last_print, last_label = now, output["label"]
    except KeyboardInterrupt:
        print("Test temps reel arrete.")
    if total:
        print(f"Replay termine : {correct / total:.1%} de fenetres correctes (transitions comprises).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test temps reel du modele EMG a 3 gestes.")
    parser.add_argument("--model_path", type=str, default=MODEL_PATH, help="Modele .json (defaut : le plus recent)")
    parser.add_argument("--vote-window", "--majority-vote", type=int, default=config.VOTE_WINDOW,
                        help="Taille du vote majoritaire en fenetres (1 = desactive)")
    parser.add_argument("--no-adapt", action="store_true", help="Desactive le suivi de derive")
    parser.add_argument("--replay", type=int, default=None, help="Rejoue dataset/S<N> au lieu de lire le capteur")
    parser.add_argument("--speed", type=float, default=1.0, help="Vitesse du replay")
    args = parser.parse_args()
    run_realtime_test(
        args.model_path,
        vote_window=args.vote_window,
        adapt=not args.no_adapt,
        replay=args.replay,
        speed=args.speed,
    )
