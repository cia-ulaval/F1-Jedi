"""Briques communes des scripts de test sur capteur (collect/code_etienne/live/).

Chaque script :
  - demarre le streamer avec prepare_streamer() et l'arrete dans un finally ;
  - lit le capteur avec le WindowReader du pont de jeu (meme code qu'en jeu) ;
  - enregistre TOUT le signal brut + l'horodatage + le geste demande dans
    dataset/live/<date>_<nom>/emg.csv, pour pouvoir le rejouer hors ligne.
"""

import csv
import datetime
import json
import sys
import time
from pathlib import Path

import numpy as np

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from config import Config  # noqa: E402
from realtime_test_etienne import WindowReader  # noqa: E402

config = Config()
LIVE_DIR = config.DATA_PRE_PATH / "live"
NO_CUE = -1          # etiquette des echantillons hors consigne (installation, calibration...)

try:
    import winsound
except ImportError:      # hors Windows : pas de bip
    winsound = None


# ---------------------------------------------------------------- dossiers --
def new_run_dir(name):
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = LIVE_DIR / f"{stamp}_{name}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def latest_run_dir(name):
    runs = sorted(LIVE_DIR.glob(f"*_{name}"))
    if not runs:
        raise FileNotFoundError(f"Aucun enregistrement *_{name} dans {LIVE_DIR}")
    return runs[-1]


def save_json(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value))


# ------------------------------------------------------------------ cues --
class Cue:
    """Etat courant de la consigne, lu par l'enregistreur a chaque paquet."""

    def __init__(self):
        self.label = NO_CUE
        self.phase = "setup"
        self.onset = time.perf_counter()

    def set(self, label, phase):
        self.label, self.phase, self.onset = int(label), phase, time.perf_counter()


def beep(kind="go"):
    if winsound is None:
        return
    try:
        freq, ms = {"go": (880, 120), "rest": (440, 120), "tick": (660, 40), "end": (330, 400)}[kind]
        winsound.Beep(freq, ms)
    except RuntimeError:
        pass


def say(text):
    print(text, flush=True)


# --------------------------------------------------------- enregistrement --
class RecordingReader(WindowReader):
    """WindowReader du jeu + copie de chaque echantillon recu.

    Lit le tampon partage complet a chaque scrutation : tant qu'on scrute plus
    souvent qu'une fois par seconde, aucun echantillon n'est perdu. Les trous
    (plus d'echantillons arrives que la taille du tampon) sont comptes.
    """

    def __init__(self, odh, window_size, window_inc, cue, buffer_size=None):
        super().__init__(odh, window_size, window_inc)
        self.cue = cue
        self.buffer_size = int(buffer_size or config.EMG_FS)
        self.t0 = time.perf_counter()
        self._clear_tape()

    def _clear_tape(self):
        self._last_total = 0
        self.chunks = []          # (t_lecture, total, echantillons chronologiques, cue, phase)
        self.lost = 0
        self.reads = []           # (t_lecture, total) a chaque lecture : sert a mesurer la regularite

    def reset(self):
        super().reset()
        self.t0 = time.perf_counter()
        self._clear_tape()

    def poll(self):
        window = super().poll()
        if window is not None:
            # Index (depuis le reset) de l'echantillon qui suit la fin de la fenetre.
            self.last_end = self._next_count - self.window_inc
        return window

    def _read(self, num_samples):
        values, count = self.odh.get_data(N=self.buffer_size)
        total = int(np.asarray(count["emg"]).reshape(-1)[0])
        data = values["emg"]
        now = time.perf_counter() - self.t0
        self.reads.append((now, total))
        new = total - self._last_total
        if new > 0:
            if new > len(data):
                self.lost += new - len(data)
                new = len(data)
            fresh = np.asarray(data[:new, 0], dtype=float)[::-1]
            self.chunks.append((now, total, fresh, self.cue.label, self.cue.phase))
            self._last_total = total
        return data[:num_samples], total

    # -- export ----------------------------------------------------------
    def tape(self):
        """Signal continu : (index, t_s, emg_V, cue, phase) par echantillon."""
        rows_idx, rows_t, rows_v, rows_cue, rows_phase = [], [], [], [], []
        fs = float(config.EMG_FS)
        for t_read, total, fresh, cue, phase in self.chunks:
            n = len(fresh)
            idx = np.arange(total - n, total)
            # Le dernier echantillon du paquet arrive a peu pres au moment de la lecture.
            rows_idx.append(idx)
            rows_t.append(t_read - (total - 1 - idx) / fs)
            rows_v.append(fresh)
            rows_cue.append(np.full(n, cue))
            rows_phase.extend([phase] * n)
        if not rows_idx:
            return np.empty(0, int), np.empty(0), np.empty(0), np.empty(0, int), []
        return (np.concatenate(rows_idx), np.concatenate(rows_t), np.concatenate(rows_v),
                np.concatenate(rows_cue), rows_phase)

    def save(self, run_dir):
        idx, t, v, cue, phase = self.tape()
        path = Path(run_dir) / "emg.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["sample", "t_s", "emg_V", "cue", "phase"])
            for row in zip(idx, t, v, cue, phase):
                writer.writerow([int(row[0]), f"{row[1]:.6f}", repr(float(row[2])), int(row[3]), row[4]])
        return path


def load_tape(run_dir):
    """Relit dataset/live/<run>/emg.csv -> dict de tableaux numpy."""
    path = Path(run_dir) / "emg.csv"
    with path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return {
        "sample": np.array([int(r["sample"]) for r in rows]),
        "t": np.array([float(r["t_s"]) for r in rows]),
        "emg": np.array([float(r["emg_V"]) for r in rows]),
        "cue": np.array([int(r["cue"]) for r in rows]),
        "phase": np.array([r["phase"] for r in rows]),
    }


class DecoderLog:
    """Sortie du decodeur a chaque fenetre, avec la consigne du moment."""

    FIELDS = ["t_s", "cue", "phase", "since_cue_s", "label", "raw_label", "rms_uV", "threshold_uV", "rest_uV"]

    def __init__(self):
        self.rows = []

    def add(self, t, cue, output):
        self.rows.append({
            "t_s": t, "cue": cue.label, "phase": cue.phase, "since_cue_s": time.perf_counter() - cue.onset,
            "label": output["label"], "raw_label": output["raw_label"], "rms_uV": output["rms_uV"],
            "threshold_uV": output["threshold_uV"], "rest_uV": output["rest_uV"],
        })

    def save(self, path):
        with Path(path).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.FIELDS)
            writer.writeheader()
            writer.writerows(self.rows)


def load_decoder_log(path):
    with Path(path).open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key in ("t_s", "since_cue_s", "rms_uV", "threshold_uV", "rest_uV"):
            row[key] = float(row[key])
        for key in ("cue", "label", "raw_label"):
            row[key] = int(row[key])
    return rows


# ------------------------------------------------------------- capteur --
class Sensor:
    """Contexte : demarre le streamer, attend les donnees, l'arrete toujours.

        with Sensor(cue) as reader:
            ...
    """

    def __init__(self, cue, device_name=None, timeout_s=None):
        self.cue = cue
        self.device_name = device_name
        self.timeout_s = config.TIMEOUT if timeout_s is None else timeout_s
        self.streamer = None
        self.reader = None
        self.connect_s = None

    def __enter__(self):
        import collect_etienne
        if self.device_name:
            collect_etienne.config.DEVICE_NAME = self.device_name
        start = time.perf_counter()
        say(f"Connexion au capteur {collect_etienne.config.DEVICE_NAME}...")
        self.streamer, odh = collect_etienne.prepare_streamer(streaming=False, return_streamer=True)
        try:
            self.reader = RecordingReader(odh, config.WINDOW_SIZE, config.WINDOW_INC, self.cue)
            if not self.reader.wait_for_emg(self.timeout_s):
                raise RuntimeError(f"Aucun echantillon EMG apres {self.timeout_s:.0f} s "
                                   f"(nom du capteur '{collect_etienne.config.DEVICE_NAME}' ?)")
        except BaseException:
            self.__exit__(None, None, None)
            raise
        self.connect_s = time.perf_counter() - start
        say(f"Capteur connecte en {self.connect_s:.1f} s.")
        self.reader.reset()
        return self.reader

    def __exit__(self, *exc):
        import collect_etienne
        if self.streamer is not None:
            say("Arret du streamer...")
            collect_etienne.stop_streamer(self.streamer)
            say(f"Streamer arrete (processus vivant : {self.streamer.is_alive()}).")
        return False


# ---------------------------------------------------------- consignes --
def run_for(reader, duration_s, on_window=None):
    """Scrute le capteur pendant duration_s ; appelle on_window(t, fenetre)."""
    deadline = time.perf_counter() + duration_s
    while time.perf_counter() < deadline:
        window = reader.poll()
        if window is not None and on_window is not None:
            on_window(time.perf_counter() - reader.t0, window)
        else:
            time.sleep(config.DELAY_BETWEEN_PREDICTIONS)


def countdown(reader, text, seconds=3, on_window=None):
    """'REPOS 3..2..1' en continuant a lire le capteur (le tampon ne deborde pas)."""
    print(text, end=" ", flush=True)
    for remaining in range(int(seconds), 0, -1):
        print(f"{remaining}..", end="", flush=True)
        beep("tick")
        run_for(reader, 1.0, on_window)
    print(flush=True)


def hold(reader, cue, label, phase, duration_s, text, on_window=None, sound="go"):
    cue.set(label, phase)
    say(f"  -> {text} pendant {duration_s:g} s")
    beep(sound)
    run_for(reader, duration_s, on_window)
