"""Coeur du pipeline EMG a 3 gestes (repos + 2 gestes actifs).

Ce module ne depend que de numpy : il ne touche ni au capteur ni a libemg,
ce qui permet de le tester hors ligne en rejouant les CSV du dataset.

Pourquoi ce decoupage en deux etages ?
Le BioPoint n'a qu'UN canal EMG. Avec un seul canal, l'information se
resume a (1) combien le muscle force (amplitude) et (2) la forme du spectre.

  Etage 1 - repos vs actif : amplitude (log-RMS) comparee a un seuil place
            entre le niveau de repos et le niveau actif. Le niveau de repos
            est suivi en continu, donc le seuil suit la derive du capteur
            (sueur, temperature de la peau).
  Etage 2 - geste A vs geste B : LDA sur des caracteristiques de FORME
            (autocorrelation normalisee). Elles ne dependent pas de
            l'amplitude, donc ni de la force de contraction, ni du gain
            electrode-peau qui change avec la sueur et la fatigue.
"""

import json
from collections import deque
from pathlib import Path

import numpy as np

MODEL_FORMAT = "f1jedi-3gestes-v2"
EPS = 1e-12


# ---------------------------------------------------------------- features --
def frame_signal(signal, window_size, window_inc):
    """Decoupe un signal 1-D en fenetres glissantes -> (n_fenetres, window_size)."""
    signal = np.asarray(signal, dtype=float).reshape(-1)
    if signal.size < window_size:
        return np.empty((0, window_size))
    view = np.lib.stride_tricks.sliding_window_view(signal, window_size)
    return view[::window_inc]


def autocovariance(windows, n_lags):
    """Autocovariance par echantillon aux retards 0..n_lags -> (n, n_lags + 1)."""
    windows = np.atleast_2d(np.asarray(windows, dtype=float))
    centered = windows - windows.mean(axis=1, keepdims=True)
    length = windows.shape[1]
    cov = np.empty((windows.shape[0], n_lags + 1))
    cov[:, 0] = np.sum(centered * centered, axis=1) / length
    for lag in range(1, n_lags + 1):
        cov[:, lag] = np.sum(centered[:, lag:] * centered[:, :-lag], axis=1) / length
    return cov


def noise_profile(rest_windows, n_lags):
    """Autocovariance moyenne du bruit de repos (retards 0..n_lags)."""
    cov = autocovariance(rest_windows, n_lags)
    if cov.shape[0] == 0:
        return np.zeros(n_lags + 1)
    return cov.mean(axis=0)


def extract_features(windows, n_lags, noise=None):
    """Retourne (log_rms, shape) pour un lot de fenetres (n, window_size).

    log_rms : (n,)  logarithme naturel de l'ecart-type de la fenetre.
    shape   : (n, n_lags) autocorrelation normalisee aux retards 1..n_lags.
              Multiplier le signal par une constante ne change pas ces valeurs.
    noise   : profil du bruit de repos (voir noise_profile). S'il est fourni,
              il est soustrait avant normalisation : la forme decrit alors la
              contraction seule, meme si le bruit de fond monte (mauvais
              contact, sueur).
    """
    cov = autocovariance(windows, n_lags)
    log_rms = 0.5 * np.log(cov[:, 0] + EPS)
    if noise is not None:
        clean = cov - np.asarray(noise, dtype=float)
        # Plancher : on ne retire jamais plus de 80 % de la puissance.
        clean[:, 0] = np.maximum(clean[:, 0], 0.2 * cov[:, 0])
        cov = clean
    shape = cov[:, 1:] / (cov[:, :1] + EPS)
    return log_rms, shape


def _sigmoid(value):
    return 1.0 / (1.0 + np.exp(-np.clip(value, -60.0, 60.0)))


# ------------------------------------------------------------------- model --
class ThreeGestureModel:
    """Parametres appris. Classes : 0 = repos, 1 = geste A, 2 = geste B."""

    def __init__(self):
        self.gestures = []          # [repos, geste A, geste B]
        self.fs = 0
        self.window_size = 0
        self.window_inc = 0
        self.n_lags = 0
        self.rest_level = 0.0       # log-RMS median au repos
        self.active_level = 0.0     # log-RMS moyen des deux gestes actifs
        self.on_frac = 0.45
        self.off_frac = 0.30
        self.mean_a = None
        self.mean_b = None
        self.weights = None         # score > 0 -> geste B
        self.noise = None           # profil du bruit de repos (retards 0..n_lags)
        self.info = {}

    # -- seuils d'activation --------------------------------------------
    @staticmethod
    def thresholds(rest_level, active_level, on_frac, off_frac, min_ratio=1.8):
        # L'ecart est borne par le bas : le seuil reste toujours au moins a
        # `min_ratio` fois le RMS de repos, sinon le bruit declenche un geste.
        span = max(active_level - rest_level, np.log(min_ratio) / on_frac)
        return rest_level + on_frac * span, rest_level + off_frac * span

    def fit(self, log_rms, shape, labels, noise, shrinkage=1e-6):
        """Apprend les seuils et la LDA. labels : 0 repos, 1 geste A, 2 geste B.

        `shape` doit deja etre compense du bruit de repos de sa session ;
        `noise` est le profil de bruit utilise au demarrage du temps reel.
        """
        self.noise = np.asarray(noise, dtype=float)
        log_rms = np.asarray(log_rms, dtype=float)
        shape = np.asarray(shape, dtype=float)
        labels = np.asarray(labels)
        for cls in (0, 1, 2):
            if not np.any(labels == cls):
                raise ValueError(f"Aucune fenetre pour la classe {cls} : impossible d'entrainer.")

        self.rest_level = float(np.median(log_rms[labels == 0]))
        self.active_level = float(0.5 * (np.mean(log_rms[labels == 1]) + np.mean(log_rms[labels == 2])))
        if self.active_level - self.rest_level < np.log(2.0):
            raise ValueError(
                "Les gestes actifs sont a peine plus forts que le repos (rapport RMS < 2). "
                "Verifie la position du capteur et recommence la collecte."
            )
        thr_on, _ = self.thresholds(self.rest_level, self.active_level, self.on_frac, self.off_frac)

        # La LDA n'apprend que sur les fenetres reellement contractees.
        keep_a = (labels == 1) & (log_rms > thr_on)
        keep_b = (labels == 2) & (log_rms > thr_on)
        if keep_a.sum() < 10 or keep_b.sum() < 10:
            raise ValueError("Pas assez de fenetres actives au-dessus du seuil pour entrainer l'etage 2.")
        xa, xb = shape[keep_a], shape[keep_b]
        self.mean_a, self.mean_b = xa.mean(axis=0), xb.mean(axis=0)
        cov = 0.5 * (np.cov(xa, rowvar=False) + np.cov(xb, rowvar=False))
        cov = np.atleast_2d(cov)
        # Regularisation minuscule, juste pour la stabilite numerique : sur les
        # donnees S1/S2 une regularisation plus forte degrade le transfert
        # entre sessions (0.85 -> 0.72 sur geste A vs geste B).
        cov = (1.0 - shrinkage) * cov + shrinkage * (np.trace(cov) / cov.shape[0]) * np.eye(cov.shape[0])
        self.weights = np.linalg.solve(cov, self.mean_b - self.mean_a)
        self.info["kept_windows"] = {"a": int(keep_a.sum()), "b": int(keep_b.sum())}
        return self

    def gesture_score(self, shape, shift=None):
        """Score LDA : positif -> geste B, negatif -> geste A."""
        midpoint = 0.5 * (self.mean_a + self.mean_b)
        if shift is not None:
            midpoint = midpoint + shift
        return (np.atleast_2d(shape) - midpoint) @ self.weights

    def predict(self, log_rms, shape):
        """Prediction fenetre par fenetre, sans memoire (sert a l'evaluation)."""
        thr_on, _ = self.thresholds(self.rest_level, self.active_level, self.on_frac, self.off_frac)
        active = np.asarray(log_rms) > thr_on
        gesture = np.where(self.gesture_score(shape) > 0, 2, 1)
        return np.where(active, gesture, 0)

    # -- persistance (JSON : lisible, pas de pickle) ----------------------
    def save(self, path):
        payload = {
            "format": MODEL_FORMAT,
            "gestures": list(self.gestures),
            "fs": self.fs,
            "window_size": self.window_size,
            "window_inc": self.window_inc,
            "n_lags": self.n_lags,
            "rest_level": self.rest_level,
            "active_level": self.active_level,
            "on_frac": self.on_frac,
            "off_frac": self.off_frac,
            "mean_a": np.asarray(self.mean_a).tolist(),
            "mean_b": np.asarray(self.mean_b).tolist(),
            "weights": np.asarray(self.weights).tolist(),
            "noise": np.asarray(self.noise).tolist(),
            "info": self.info,
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path):
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("format") != MODEL_FORMAT:
            raise ValueError(f"{path} n'est pas un modele {MODEL_FORMAT}.")
        model = cls()
        for key in ("gestures", "fs", "window_size", "window_inc", "n_lags",
                    "rest_level", "active_level", "on_frac", "off_frac", "info"):
            setattr(model, key, payload[key])
        model.mean_a = np.asarray(payload["mean_a"], dtype=float)
        model.mean_b = np.asarray(payload["mean_b"], dtype=float)
        model.weights = np.asarray(payload["weights"], dtype=float)
        model.noise = np.asarray(payload["noise"], dtype=float)
        return model


# ----------------------------------------------------------------- decoder --
class OnlineDecoder:
    """Decodage temps reel avec memoire : hysteresis, vote, suivi de derive.

    Appeler `update(window)` a chaque nouvelle fenetre. Le resultat contient
    `label` (0/1/2, lisse) et `name` (nom du geste).
    """

    def __init__(self, model, vote_window=9, adapt=True,
                 rest_tau_s=5.0, active_tau_s=15.0, shift_tau_s=15.0,
                 max_rest_drift=np.log(4.0), max_shift_frac=1.5, onset_windows=0):
        self.model = model
        self.vote_window = max(1, int(vote_window))
        # Attaque d'une contraction : pendant ces fenetres on ne choisit pas
        # encore de geste (sortie repos, score remis a jour a chaque fenetre).
        self.onset_windows = max(0, int(onset_windows))
        self.adapt = adapt
        step_s = model.window_inc / float(model.fs)
        self._rest_alpha = step_s / rest_tau_s
        self._active_alpha = step_s / active_tau_s
        self._shift_alpha = step_s / shift_tau_s
        self._max_rest_drift = max_rest_drift
        # Le decalage de forme est borne (en fraction de la demi-distance entre
        # les deux gestes) pour que l'adaptation ne parte pas a la derive.
        self._max_shift = max_shift_frac * 0.5 * float(np.linalg.norm(model.mean_b - model.mean_a))
        self.reset()

    def reset(self):
        self.rest_level = float(self.model.rest_level)
        self.active_level = float(self.model.active_level)
        self._rest_anchor = self.rest_level
        self.shift = np.zeros_like(self.model.mean_a)
        self.noise = np.array(self.model.noise, dtype=float)
        self.is_active = False
        self._score = 0.0
        self._active_run = 0
        self._votes = deque(maxlen=self.vote_window)
        self.label = 0

    # -- calibration ------------------------------------------------------
    def calibrate_rest(self, windows, max_ratio=3.0):
        """Recale le niveau de repos sur des fenetres mesurees bras detendu.

        Retourne le niveau mesure (log-RMS). S'il depasse `max_ratio` fois le
        repos appris, la personne n'etait sans doute pas detendue (ou le
        contact est mauvais) : on plafonne pour ne pas rendre le decodeur sourd.
        """
        log_rms, _ = extract_features(windows, self.model.n_lags)
        if log_rms.size == 0:
            return self.rest_level
        raw_measure = float(np.median(log_rms))
        measured = min(raw_measure, self.model.rest_level + float(np.log(max_ratio)))
        if raw_measure <= measured:
            self.noise = noise_profile(windows, self.model.n_lags)
        # L'ecart repos -> actif appris est conserve : seul le zero bouge.
        self.active_level += measured - self.rest_level
        self.rest_level = measured
        self._rest_anchor = measured
        return raw_measure

    def recalibrate(self, rest_windows, a_windows, b_windows, blend=1.0):
        """Recalibration rapide : quelques secondes de chaque geste.

        Recale les niveaux d'amplitude et les centres des deux gestes en
        gardant la direction LDA apprise. `blend` = poids des nouvelles mesures.
        """
        n_lags = self.model.n_lags
        noise = noise_profile(rest_windows, n_lags)
        rest_rms, _ = extract_features(rest_windows, n_lags)
        a_rms, a_shape = extract_features(a_windows, n_lags, noise)
        b_rms, b_shape = extract_features(b_windows, n_lags, noise)
        if min(rest_rms.size, a_rms.size, b_rms.size) == 0:
            raise ValueError("Recalibration : il manque des fenetres pour au moins un geste.")
        rest_level = float(np.median(rest_rms))
        active_level = float(0.5 * (np.mean(a_rms) + np.mean(b_rms)))
        if active_level - rest_level < np.log(2.0):
            raise ValueError("Recalibration refusee : gestes trop faibles par rapport au repos.")
        thr_on, _ = self.model.thresholds(rest_level, active_level, self.model.on_frac, self.model.off_frac)
        a_shape, b_shape = a_shape[a_rms > thr_on], b_shape[b_rms > thr_on]
        if min(len(a_shape), len(b_shape)) < 5:
            raise ValueError("Recalibration refusee : contractions trop courtes.")
        new_mid = 0.5 * (a_shape.mean(axis=0) + b_shape.mean(axis=0))
        old_mid = 0.5 * (self.model.mean_a + self.model.mean_b)
        # Garde-fou : apres recalage, les deux gestes doivent encore tomber
        # du bon cote de la frontiere, sinon on ne touche a rien.
        if (a_shape.mean(axis=0) - new_mid) @ self.model.weights >= 0 or \
           (b_shape.mean(axis=0) - new_mid) @ self.model.weights <= 0:
            raise ValueError("Recalibration refusee : les deux gestes ne se distinguent plus. Refais une collecte.")
        self.rest_level = rest_level
        self._rest_anchor = rest_level
        self.active_level = active_level
        self.noise = noise
        self.shift = (1.0 - blend) * self.shift + blend * (new_mid - old_mid)
        self._votes.clear()
        self.is_active = False
        self.label = 0
        return {"rest_uV": float(np.exp(rest_level) * 1e6), "active_uV": float(np.exp(active_level) * 1e6)}

    # -- boucle temps reel ----------------------------------------------
    def update(self, window):
        model = self.model
        cov = autocovariance(window, model.n_lags)
        log_rms, shape = extract_features(window, model.n_lags, self.noise)
        log_rms, shape = float(log_rms[0]), shape[0]
        thr_on, thr_off = model.thresholds(self.rest_level, self.active_level, model.on_frac, model.off_frac)

        was_active = self.is_active
        if self.is_active:
            self.is_active = log_rms > thr_off
        else:
            self.is_active = log_rms > thr_on

        raw_score = float(model.gesture_score(shape, self.shift)[0])
        if self.is_active:
            if not was_active:
                self._active_run = 0
            self._active_run += 1
            if self._active_run <= max(1, self.onset_windows):
                # Debut de contraction : le score repart de la fenetre courante.
                self._score = raw_score
            else:
                # Accumule l'evidence pendant la contraction (constante ~100 ms).
                self._score += 0.2 * (raw_score - self._score)
            if self._active_run <= self.onset_windows:
                raw_label = 0
            else:
                raw_label = 2 if self._score > 0 else 1
        else:
            self._active_run = 0
            raw_label = 0

        if self.adapt:
            self._adapt(log_rms, shape, raw_label, thr_off, cov[0])

        self._votes.append(raw_label)
        counts = np.bincount(self._votes, minlength=3)
        best = int(np.argmax(counts))
        # On ne change de sortie que sur une majorite stricte : pas de
        # clignotement quand deux gestes sont a egalite.
        if best != self.label and counts[best] > counts[self.label]:
            self.label = best

        span = max(thr_on - thr_off, 1e-6)
        p_active = float(_sigmoid((log_rms - 0.5 * (thr_on + thr_off)) / span))
        p_b = float(_sigmoid(self._score if self.is_active else raw_score))
        return {
            "label": self.label,
            "name": model.gestures[self.label],
            "raw_label": raw_label,
            "probabilities": [1.0 - p_active, p_active * (1.0 - p_b), p_active * p_b],
            "rms_uV": float(np.exp(log_rms) * 1e6),
            "threshold_uV": float(np.exp(thr_on) * 1e6),
            "rest_uV": float(np.exp(self.rest_level) * 1e6),
        }

    def _adapt(self, log_rms, shape, raw_label, thr_off, cov):
        if raw_label == 0:
            # Suivi du bruit de repos, seulement quand on est nettement au repos.
            if log_rms < thr_off:
                self.noise += self._rest_alpha * (cov - self.noise)
                self.rest_level += self._rest_alpha * (log_rms - self.rest_level)
                self.rest_level = float(np.clip(self.rest_level,
                                                self._rest_anchor - self._max_rest_drift,
                                                self._rest_anchor + self._max_rest_drift))
            return
        # Contraction installee depuis 200 ms : on ignore les transitions.
        if self._active_run * self.model.window_inc / float(self.model.fs) < 0.2:
            return
        self.active_level += self._active_alpha * (log_rms - self.active_level)
        # Derive commune aux deux gestes (fatigue : le spectre glisse vers le
        # bas pour tous les gestes) : on deplace le point milieu, pas la LDA.
        centre = self.model.mean_b if raw_label == 2 else self.model.mean_a
        target = shape - centre
        self.shift += self._shift_alpha * (target - self.shift)
        norm = float(np.linalg.norm(self.shift))
        if norm > self._max_shift:
            self.shift *= self._max_shift / norm


# ----------------------------------------------------------------- dataset --
def load_session(session_dir, gestures):
    """Charge une session collectee avec le GUI libemg.

    Les fichiers s'appellent C_<classe>_R_<rep>_emg.csv ; la correspondance
    classe -> nom du geste est lue dans collection_details.json, donc une
    ancienne session a 5 gestes reste utilisable : on n'en garde que les 3
    gestes demandes, par NOM.

    Retourne une liste de dicts {label, rep, signal}, label dans 0/1/2.
    """
    session_dir = Path(session_dir)
    details_path = session_dir / "collection_details.json"
    if not details_path.exists():
        raise FileNotFoundError(f"collection_details.json introuvable dans {session_dir}")
    details = json.loads(details_path.read_text(encoding="utf-8"))
    class_map = {int(index): name for index, name in details["class_map"].items()}
    name_to_index = {name: index for index, name in class_map.items()}
    missing = [name for name in gestures if name not in name_to_index]
    if missing:
        raise ValueError(f"{session_dir.name} ne contient pas les gestes {missing} (disponibles : {sorted(name_to_index)}).")

    recordings = []
    for label, name in enumerate(gestures):
        files = sorted(session_dir.glob(f"C_{name_to_index[name]}_R_*_emg.csv"))
        if not files:
            raise FileNotFoundError(f"Aucun fichier pour {name} dans {session_dir}")
        for file in files:
            rep = int(file.stem.split("_")[3])
            signal = np.loadtxt(file, delimiter=",", ndmin=2)[:, 0]
            recordings.append({"label": label, "rep": rep, "signal": signal, "file": file.name})
    return recordings
