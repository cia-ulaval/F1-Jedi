"""Etape 2 - qualite du signal : N tours de repos / geste A / geste B (config.GESTURES).

    python collect/code_etienne/live/live_signal_quality.py --rounds 3 --hold 4
    python collect/code_etienne/live/live_signal_quality.py --replay dataset/live/<run>   # reanalyse

Affiche le RMS median (uV) par geste, le rapport actif/repos, et la
separabilite des deux gestes dans les features de l'etage 2 (forme).
Cibles : repos < 15 uV, rapport >= 5x pour les deux gestes, gestes separables.
Resultats : dataset/live/<date>_quality/{emg.csv, summary.json}.
"""

import argparse

import numpy as np

from live_common import (NO_CUE, Cue, Sensor, config, countdown, hold, load_tape, new_run_dir, run_for,
                         save_json, say)
from emg_core import ThreeGestureModel, extract_features, frame_signal, noise_profile

SKIP_S = 0.5            # debut de chaque pose ignore (le geste s'installe)
TARGET_REST_UV = 15.0
TARGET_RATIO = 5.0      # repere ; la decision se fait sur le decodeur rejoue
TARGET_REPLAY = 0.95    # temps reel simule, un tour laisse de cote
TARGET_ABOVE = 0.95     # part des fenetres du geste le plus faible au-dessus du seuil


def segments(tape):
    """Plages contigues (cue, phase) du flux enregistre."""
    key = np.array([f"{c}|{p}" for c, p in zip(tape["cue"], tape["phase"])])
    cuts = np.flatnonzero(key[1:] != key[:-1]) + 1
    bounds = np.concatenate([[0], cuts, [len(key)]])
    return [{"label": int(tape["cue"][a]), "phase": str(tape["phase"][a]), "start": int(a), "end": int(b)}
            for a, b in zip(bounds[:-1], bounds[1:])]


def hold_windows(tape, skip_s=SKIP_S):
    """{label: [fenetres d'une pose, ...]} pour les phases 'hold'."""
    skip = int(skip_s * config.EMG_FS)
    out = {0: [], 1: [], 2: []}
    for seg in segments(tape):
        if seg["phase"] != "hold" or seg["label"] not in out:
            continue
        windows = frame_signal(tape["emg"][seg["start"] + skip:seg["end"]], config.WINDOW_SIZE, config.WINDOW_INC)
        if len(windows):
            out[seg["label"]].append(windows)
    return out


def latest_model():
    from realtime_test_etienne import find_last_model
    try:
        path = find_last_model()
    except FileNotFoundError:
        return None, None
    return ThreeGestureModel.load(path), path


def analyse(tape):
    names = config.GESTURES
    held = hold_windows(tape)
    if not all(held[label] for label in (0, 1, 2)):
        raise ValueError("Il manque des poses pour au moins un geste.")
    rest_all = np.vstack(held[0])
    noise = noise_profile(rest_all, config.N_LAGS)

    result = {"gestures": {}}
    levels = {}
    for label in (0, 1, 2):
        rms = [np.exp(extract_features(w, config.N_LAGS)[0]) * 1e6 for w in held[label]]
        per_round = [float(np.median(r)) for r in rms]
        levels[label] = float(np.median(np.concatenate(rms)))
        result["gestures"][names[label]] = {"median_rms_uV": levels[label], "per_round_uV": per_round}
    for label in (1, 2):
        result["gestures"][names[label]]["ratio_vs_rest"] = levels[label] / levels[0]

    # Etage 2 : on entraine le meme modele que trainning_etienne.py, en
    # laissant un tour de cote a chaque fois (validation croisee par tour).
    n_rounds = min(len(held[label]) for label in (0, 1, 2))
    fold_acc = []
    for test_round in range(n_rounds):
        train = {label: [w for k, w in enumerate(held[label]) if k != test_round] for label in (0, 1, 2)}
        model = _fit(train, noise)
        if model is None:
            continue
        hits = total = 0
        for label in (1, 2):
            log_rms, shape = extract_features(held[label][test_round], config.N_LAGS, noise)
            thr_on, _ = model.thresholds(model.rest_level, model.active_level, model.on_frac, model.off_frac)
            active = log_rms > thr_on
            pred = np.where(model.gesture_score(shape[active]) > 0, 2, 1)
            hits += int(np.sum(pred == label))
            total += int(active.sum())
        fold_acc.append(hits / max(total, 1))
    result["stage2_cv_accuracy_per_round"] = fold_acc
    result["stage2_cv_accuracy"] = float(np.mean(fold_acc)) if fold_acc else None

    model = _fit(held, noise)
    if model is not None:
        result["stage2_fisher"] = _fisher(model, held, noise)
    # Transfert : le dernier modele entraine (anciennes sessions) separe-t-il ces gestes ?
    old, path = latest_model()
    if old is not None:
        acc = []
        for label in (1, 2):
            log_rms, shape = extract_features(np.vstack(held[label]), config.N_LAGS, noise)
            keep = log_rms > np.log(2.0 * levels[0] * 1e-6)
            centre_shift = None
            pred = np.where(old.gesture_score(shape[keep], centre_shift) > 0, 2, 1)
            acc.append(float(np.mean(pred == label)) if keep.any() else None)
        result["latest_model"] = {"path": str(path), "stage2_accuracy_per_gesture": acc}

    result["decoder"] = decoder_check(tape)

    ok_rest = levels[0] < TARGET_REST_UV
    ok_ratio = all(result["gestures"][names[l]]["ratio_vs_rest"] >= TARGET_RATIO for l in (1, 2))
    ok_sep = result["stage2_cv_accuracy"] is not None and result["stage2_cv_accuracy"] >= 0.90
    dec = result["decoder"]
    ok_decoder = dec["replay_accuracy"] >= TARGET_REPLAY and dec["weakest_above_threshold"] >= TARGET_ABOVE
    result["targets"] = {"rest_below_15uV": ok_rest, "ratio_at_least_5x": ok_ratio, "stage2_separable_90pct": ok_sep,
                         "decoder_replay_95pct": ok_decoder}
    # Le rapport 5x est un repere. Ce qui decide : le decodeur reel rejoue sur
    # ces poses (mesure : Hand_Close a 4.0x -> 99.6 % en temps reel simule).
    result["ready_to_collect"] = bool(ok_rest and ok_sep and ok_decoder)
    return result


def decoder_check(tape):
    """Le decodeur du jeu, entraine sur tous les tours sauf un, teste sur celui-la."""
    import trainning_etienne as tr
    recs, count = [], {0: 0, 1: 0, 2: 0}
    for seg in segments(tape):
        if seg["phase"] == "hold" and seg["label"] in count:
            recs.append({"label": seg["label"], "rep": count[seg["label"]], "session": 0,
                         "signal": tape["emg"][seg["start"]:seg["end"]]})
            count[seg["label"]] += 1
    n_rounds = min(count.values())
    confusion = np.zeros((3, 3), dtype=int)
    above = {1: [], 2: []}
    for k in range(n_rounds):
        try:
            model = tr.fit_model([r for r in recs if r["rep"] != k])
        except ValueError as error:
            say(f"  (decodeur non entrainable sans le tour {k + 1} : {error})")
            continue
        thr_on, _ = model.thresholds(model.rest_level, model.active_level, model.on_frac, model.off_frac)
        for rec in recs:
            if rec["rep"] == k and rec["label"] in above:
                skip = int(SKIP_S * config.EMG_FS)
                log_rms, _ = extract_features(frame_signal(rec["signal"][skip:], config.WINDOW_SIZE, config.WINDOW_INC),
                                              config.N_LAGS)
                above[rec["label"]].append(float(np.mean(log_rms > thr_on)))
        confusion += tr.replay_accuracy(model, [r for r in recs if r["rep"] == k])
    total = max(confusion.sum(), 1)
    frac = {config.GESTURES[l]: float(np.mean(v)) if v else 0.0 for l, v in above.items()}
    return {"replay_accuracy": float(np.trace(confusion) / total), "confusion": confusion.tolist(),
            "above_threshold": frac, "weakest_above_threshold": min(frac.values())}


def _fit(held, noise):
    log_rms, shape, labels = [], [], []
    for label in (0, 1, 2):
        for windows in held[label]:
            r, s = extract_features(windows, config.N_LAGS, noise)
            log_rms.append(r), shape.append(s), labels.append(np.full(len(r), label))
    model = ThreeGestureModel()
    model.gestures, model.fs = list(config.GESTURES), config.EMG_FS
    model.window_size, model.window_inc, model.n_lags = config.WINDOW_SIZE, config.WINDOW_INC, config.N_LAGS
    try:
        return model.fit(np.concatenate(log_rms), np.vstack(shape), np.concatenate(labels), noise)
    except ValueError as error:
        say(f"  (modele non entrainable : {error})")
        return None


def _fisher(model, held, noise):
    scores = {}
    for label in (1, 2):
        log_rms, shape = extract_features(np.vstack(held[label]), config.N_LAGS, noise)
        scores[label] = model.gesture_score(shape)
    gap = abs(np.mean(scores[2]) - np.mean(scores[1]))
    spread = np.sqrt(0.5 * (np.var(scores[1]) + np.var(scores[2])))
    return float(gap / max(spread, 1e-12))


def report(result):
    say("\n=== Qualite du signal ===")
    for name, info in result["gestures"].items():
        ratio = f"  rapport {info['ratio_vs_rest']:.1f}x" if "ratio_vs_rest" in info else ""
        rounds = ", ".join(f"{v:.1f}" for v in info["per_round_uV"])
        say(f"  {name:<16} RMS median {info['median_rms_uV']:6.1f} uV{ratio}   (par tour : {rounds})")
    cv = result["stage2_cv_accuracy"]
    say(f"  Etage 2 ({config.GESTURES[1]} vs {config.GESTURES[2]}), validation par tour : "
        f"{'n/a' if cv is None else f'{cv:.1%}'}  {['%.0f%%' % (100 * a) for a in result['stage2_cv_accuracy_per_round']]}")
    if "stage2_fisher" in result:
        say(f"  Separation de Fisher (ecart/dispersion du score LDA) : {result['stage2_fisher']:.2f}  (> 2 = bien separe)")
    dec = result["decoder"]
    above = ", ".join(f"{name} {value:.0%}" for name, value in dec["above_threshold"].items())
    say(f"  Decodeur du jeu rejoue (un tour laisse de cote) : {dec['replay_accuracy']:.1%} ; "
        f"fenetres au-dessus du seuil : {above}")
    if "latest_model" in result:
        acc = result["latest_model"]["stage2_accuracy_per_gesture"]
        say(f"  (Dernier modele sauvegarde, peut-etre d'une autre personne : "
            f"{config.GESTURES[1]} {acc[0]:.0%}, {config.GESTURES[2]} {acc[1]:.0%})")
    say(f"  Cibles : {result['targets']}")
    if result["ready_to_collect"]:
        weak = [n for n, i in result["gestures"].items() if i.get("ratio_vs_rest", 99) < TARGET_RATIO]
        note = f" (rapport < 5x pour {', '.join(weak)}, mais le decodeur le detecte bien)" if weak else ""
        say("  => PRET pour la collecte" + note)
    else:
        say("  => PAS PRET : deplacer le capteur (ou renforcer le geste faible) et refaire")


def record(rounds, hold_s, rest_s, gap_s, device):
    run_dir = new_run_dir("quality")
    cue = Cue()
    reader = None
    names = config.GESTURES
    try:
        with Sensor(cue, device_name=device) as reader:
            say(f"\n{rounds} tours : REPOS {rest_s:g} s, {names[1]} {hold_s:g} s, {names[2]} {hold_s:g} s.")
            say("Force moderee, celle que tu utiliserais en jeu. Relache completement entre les poses.\n")
            cue.set(NO_CUE, "setup")
            run_for(reader, 2.0)
            for k in range(rounds):
                say(f"--- Tour {k + 1}/{rounds}")
                cue.set(NO_CUE, "gap")
                countdown(reader, "REPOS dans", 3)
                hold(reader, cue, 0, "hold", rest_s, f"REPOS ({names[0]}), main detendue", sound="rest")
                for label in (1, 2):
                    cue.set(0, "gap")
                    countdown(reader, f"Prepare {names[label]} :", 3)
                    hold(reader, cue, label, "hold", hold_s, f"TIENS {names[label]}")
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
    parser = argparse.ArgumentParser(description="Etape 2 : qualite du signal avant la collecte.")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--hold", type=float, default=4.0, help="Duree de chaque geste actif (s)")
    parser.add_argument("--rest", type=float, default=4.0, help="Duree du repos mesure (s)")
    parser.add_argument("--gap", type=float, default=2.0, help="Pause apres chaque geste (s)")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--replay", type=str, default=None, help="Reanalyse un enregistrement dataset/live/<run>")
    args = parser.parse_args()

    run_dir = args.replay or record(args.rounds, args.hold, args.rest, args.gap, args.device)
    result = analyse(load_tape(run_dir))
    save_json(f"{run_dir}/summary.json", result)
    report(result)
    say(f"Resultats : {run_dir}")


if __name__ == "__main__":
    main()
