"""Enquete sur les gestes : poing a 3 forces + gestes candidats, avec le capteur au meme endroit.

    python collect/code_etienne/live/live_gesture_survey.py --rounds 3
    python collect/code_etienne/live/live_gesture_survey.py --replay dataset/live/<run>   # reanalyse

Questions :
  1. Le modele actuel prend-il un poing FORT pour Wrist_Extension ?
  2. Reentrainer avec des poings de forces variees regle-t-il le probleme ?
  3. Un autre geste (Hand_Open, Wrist_Flexion, pince) se separe-t-il mieux de Wrist_Extension ?
Resultats : dataset/live/<date>_survey/{emg.csv, summary.json}.
"""

import argparse

import numpy as np

from live_common import (NO_CUE, Cue, Sensor, config, countdown, hold, load_tape, new_run_dir, run_for,
                         save_json, say)
from emg_core import OnlineDecoder, ThreeGestureModel, extract_features, frame_signal, noise_profile

# (code, cle, consigne affichee)
POSES = [
    (0, "rest", "REPOS, main detendue"),
    (1, "hc_light", "POING LEGER (juste ferme, sans serrer)"),
    (2, "hc_medium", "POING MOYEN (comme en jeu)"),
    (3, "hc_strong", "POING FORT (serre fort)"),
    (4, "wrist_ext", "WRIST_EXTENSION (poignet releve, force du jeu)"),
    (5, "hand_open", "HAND_OPEN (doigts tendus et ecartes, poignet a plat)"),
    (6, "wrist_flex", "WRIST_FLEXION (poignet plie vers le bas)"),
    (7, "pinch", "PINCE (bout du pouce contre bout de l'index, appuie)"),
]
KEY = {code: key for code, key, _ in POSES}
CODE = {key: code for code, key, _ in POSES}
SKIP_S = 0.5


# ------------------------------------------------------------- analyse --
def pose_signals(tape, skip_s=SKIP_S):
    """{cle: [signal d'un tour, ...]} pour les phases 'hold'."""
    out = {key: [] for key in CODE}
    cue, phase, emg = tape["cue"], tape["phase"], tape["emg"]
    change = np.flatnonzero((cue[1:] != cue[:-1]) | (phase[1:] != phase[:-1])) + 1
    bounds = np.concatenate([[0], change, [len(emg)]])
    skip = int(skip_s * config.EMG_FS)
    for a, b in zip(bounds[:-1], bounds[1:]):
        if phase[a] == "hold" and int(cue[a]) in KEY and b - a > skip + config.WINDOW_SIZE:
            out[KEY[int(cue[a])]].append(emg[a + skip:b])
    return out


def windows(signal):
    return frame_signal(signal, config.WINDOW_SIZE, config.WINDOW_INC)


def amplitude_table(poses):
    rest = np.median(np.exp(extract_features(np.vstack([windows(s) for s in poses["rest"]]), config.N_LAGS)[0]))
    table = {}
    for key, signals in poses.items():
        if not signals:
            continue
        per_round = [float(np.median(np.exp(extract_features(windows(s), config.N_LAGS)[0])) * 1e6) for s in signals]
        table[key] = {"rms_uV": float(np.median(per_round)), "per_round_uV": per_round,
                      "ratio": float(np.median(per_round) / (rest * 1e6))}
    return table


def current_model_view(poses, model):
    """Ce que le modele actuel (sans memoire) fait de chaque pose, repos recalibre sur cette seance."""
    decoder = OnlineDecoder(model)
    decoder.calibrate_rest(np.vstack([windows(s) for s in poses["rest"]]))
    thr_on, _ = model.thresholds(decoder.rest_level, decoder.active_level, model.on_frac, model.off_frac)
    view = {}
    for key, signals in poses.items():
        if not signals:
            continue
        log_rms, shape = extract_features(np.vstack([windows(s) for s in signals]), model.n_lags, decoder.noise)
        score = model.gesture_score(shape)
        active = log_rms > thr_on
        labels = np.where(active, np.where(score > 0, 2, 1), 0)
        counts = np.bincount(labels, minlength=3) / max(len(labels), 1)
        view[key] = {"No_Motion": float(counts[0]), "Hand_Close": float(counts[1]),
                     "Wrist_Extension": float(counts[2]),
                     "lda_score_median": float(np.median(score[active])) if active.any() else None}
    return view


def trio_replay(poses, second, train_second=None):
    """Decodeur du jeu entraine sur {repos, second, wrist_ext}, un tour laisse de cote.

    second / train_second : liste de cles de poses rangees dans la classe 1.
    train_second (si fourni) sert a l'entrainement, second au test : mesure
    ce que devient un poing fort quand le modele n'a vu que des poings moyens.
    """
    import trainning_etienne as tr
    train_second = second if train_second is None else train_second
    n_rounds = min(len(poses[k]) for k in ["rest", "wrist_ext"] + list(second) + list(train_second))

    def recs(keys_1, rounds):
        out = []
        for r in rounds:
            out.append({"label": 0, "rep": r, "session": 0, "signal": poses["rest"][r]})
            for key in keys_1:
                out.append({"label": 1, "rep": r, "session": 0, "signal": poses[key][r]})
            out.append({"label": 2, "rep": r, "session": 0, "signal": poses["wrist_ext"][r]})
        return out

    confusion = np.zeros((3, 3), dtype=int)
    for k in range(n_rounds):
        try:
            model = tr.fit_model(recs(train_second, [r for r in range(n_rounds) if r != k]))
        except ValueError as error:
            return {"error": str(error)}
        confusion += tr.replay_accuracy(model, recs(second, [k]))
    recall = confusion.diagonal() / np.maximum(confusion.sum(axis=1), 1)
    return {"accuracy": float(np.trace(confusion) / max(confusion.sum(), 1)),
            "recall": {"No_Motion": float(recall[0]), "second": float(recall[1]), "Wrist_Extension": float(recall[2])},
            "second_as_wrist_ext": float(confusion[1, 2] / max(confusion[1].sum(), 1)),
            "confusion": confusion.tolist()}


def analyse(tape, model=None):
    poses = pose_signals(tape)
    result = {"rounds": {k: len(v) for k, v in poses.items()}, "amplitude": amplitude_table(poses)}
    if model is not None:
        result["current_model"] = current_model_view(poses, model)
    hc = ["hc_light", "hc_medium", "hc_strong"]
    result["hc_trained_medium_tested"] = {key: trio_replay(poses, [key], ["hc_medium"]) for key in hc}
    result["hc_trained_all_forces_tested"] = {key: trio_replay(poses, [key], hc) for key in hc}
    result["candidates"] = {key: trio_replay(poses, [key]) for key in ["hc_medium", "hand_open", "wrist_flex", "pinch"]}
    return result


def report(result):
    pct = lambda v: "  n/a" if v is None else f"{100 * v:4.0f}%"
    say("\n=== 1. Amplitude (RMS median, rapport au repos) ===")
    for key, info in result["amplitude"].items():
        rounds = ", ".join(f"{v:.0f}" for v in info["per_round_uV"])
        say(f"  {key:<11} {info['rms_uV']:7.1f} uV  {info['ratio']:5.1f}x   (tours : {rounds})")

    if "current_model" in result:
        say("\n=== 2. Modele actuel, fenetre par fenetre (repos recalibre) ===")
        say(f"  {'pose':<11} {'-> repos':>9} {'-> poing':>9} {'-> poignet':>11}")
        for key, view in result["current_model"].items():
            say(f"  {key:<11} {pct(view['No_Motion']):>9} {pct(view['Hand_Close']):>9} {pct(view['Wrist_Extension']):>11}")

    say("\n=== 3. Poing : entraine sur poing MOYEN seulement vs sur les 3 forces (temps reel simule) ===")
    say(f"  {'teste sur':<11} {'moyen seul : bien classe / pris pour poignet':<46} {'3 forces : bien classe / pris pour poignet'}")
    for key in ("hc_light", "hc_medium", "hc_strong"):
        a = result["hc_trained_medium_tested"][key]
        b = result["hc_trained_all_forces_tested"][key]
        fa = "erreur" if "error" in a else f"{pct(a['recall']['second'])} / {pct(a['second_as_wrist_ext'])}"
        fb = "erreur" if "error" in b else f"{pct(b['recall']['second'])} / {pct(b['second_as_wrist_ext'])}"
        say(f"  {key:<11} {fa:<46} {fb}")

    say("\n=== 4. Candidat pour le 2e geste (avec repos + Wrist_Extension), temps reel simule ===")
    for key, res in result["candidates"].items():
        if "error" in res:
            say(f"  {key:<11} non entrainable : {res['error']}")
            continue
        r = res["recall"]
        say(f"  {key:<11} global {pct(res['accuracy'])} | repos {pct(r['No_Motion'])}  geste {pct(r['second'])}  "
            f"poignet {pct(r['Wrist_Extension'])} | geste pris pour poignet {pct(res['second_as_wrist_ext'])}")


# ----------------------------------------------------------- enregistrement --
def record(rounds, hold_s, gap_s, device):
    run_dir = new_run_dir("survey")
    cue = Cue()
    reader = None
    try:
        with Sensor(cue, device_name=device) as reader:
            say(f"\n{rounds} tours de {len(POSES)} poses, {hold_s:g} s chacune. Lis la consigne pendant le compte a rebours.")
            say("Tiens chaque pose CONSTANTE. Relache completement entre les poses.\n")
            cue.set(NO_CUE, "setup")
            run_for(reader, 2.0)
            for k in range(rounds):
                say(f"--- Tour {k + 1}/{rounds}")
                for code, _, text in POSES:
                    cue.set(0, "gap")
                    countdown(reader, f"Prochaine : {text} ->", 3)
                    hold(reader, cue, code, "hold", hold_s, f"TIENS : {text}", sound="rest" if code == 0 else "go")
                    cue.set(0, "gap")
                    say("  -> relache")
                    run_for(reader, gap_s)
            cue.set(NO_CUE, "end")
            say("\nTermine, merci. Analyse...")
    finally:
        if reader is not None:
            reader.save(run_dir)
            say(f"Signal enregistre : {run_dir / 'emg.csv'} ({reader.lost} echantillons perdus)")
    return run_dir


def main():
    parser = argparse.ArgumentParser(description="Enquete : poing a 3 forces et gestes candidats.")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--hold", type=float, default=4.0)
    parser.add_argument("--gap", type=float, default=2.0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--model", type=str, default=None, help="Modele actuel (defaut : le plus recent)")
    parser.add_argument("--replay", type=str, default=None, help="Reanalyse dataset/live/<run>")
    args = parser.parse_args()

    run_dir = args.replay or record(args.rounds, args.hold, args.gap, args.device)
    from realtime_test_etienne import find_last_model
    model_path = args.model or str(find_last_model())
    result = analyse(load_tape(run_dir), ThreeGestureModel.load(model_path))
    result["model"] = model_path
    save_json(f"{run_dir}/summary.json", result)
    report(result)
    say(f"\nResultats : {run_dir}")


if __name__ == "__main__":
    main()
