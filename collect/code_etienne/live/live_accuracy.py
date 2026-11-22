"""Etape 5 - precision en direct : calibration du repos puis consignes aleatoires.

    python collect/code_etienne/live/live_accuracy.py --cues 15 --hold 4 --gap 2
    python collect/code_etienne/live/live_accuracy.py --model models/f1jedi_3g_S3_....json --seed 7

Le decodeur est exactement celui du jeu (OnlineDecoder + WindowReader). A chaque
fenetre on note la sortie et la consigne. Mesures : exactitude (sans les 400 ms
apres chaque consigne), matrice de confusion, latence consigne -> bonne sortie,
clignotement pendant les gestes tenus.
Cibles : >= 90 %, latence < 400 ms, clignotement ~0.
Resultats : dataset/live/<date>_accuracy/{emg.csv, decoder.csv, summary.json}.
"""

import argparse
import random

import numpy as np

from live_common import (Cue, DecoderLog, NO_CUE, Sensor, beep, config, countdown, hold, new_run_dir, run_for,
                         save_json, say)
import live_metrics
from emg_core import OnlineDecoder
from realtime_test_etienne import find_last_model, load_model

CUE_TEXT = {
    "No_Motion": "REPOS (main detendue)",
    "Hand_Close": "HAND_CLOSE (poing ferme)",
    "Wrist_Extension": "WRIST_EXTENSION (poignet releve)",
}


def calibrate(reader, decoder, cue, seconds=None):
    """Meme calibration que le pont de jeu : 0.5 s ignoree puis `seconds` de repos."""
    seconds = config.REST_CALIBRATION_S if seconds is None else seconds
    say(f"\nCalibration du repos : main DETENDUE pendant {seconds + 0.5:g} s...")
    cue.set(0, "calib_settle")
    beep("rest")
    run_for(reader, 0.5)
    cue.set(0, "calib")
    windows = []
    run_for(reader, seconds, lambda t, w: windows.append(w))
    learned = float(np.exp(decoder.model.rest_level) * 1e6)
    measured = float(np.exp(decoder.calibrate_rest(np.array(windows))) * 1e6) if windows else None
    say(f"  repos mesure {measured:.1f} uV (a l'entrainement : {learned:.1f} uV)" if measured else "  aucune donnee !")
    cue.set(0, "ready")
    return measured


def cue_order(n_cues, n_classes, rng):
    order = [k % n_classes for k in range(n_cues)]
    rng.shuffle(order)
    return order


def run_test(reader, decoder, cue, log, order, hold_s, gap_s, block=0):
    """Un bloc de consignes. Toutes les fenetres passent dans le decodeur (comme en jeu)."""
    names = decoder.model.gestures
    cue.block = block

    def on_window(t, window):
        log.add(t, cue, decoder.update(window), reader.last_end)

    say(f"\n{len(order)} consignes au hasard, {hold_s:g} s chacune, {gap_s:g} s de repos entre deux.")
    say("Reagis des que la consigne s'affiche (bip). Force du jeu, constante. Relache completement aux pauses.")
    cue.set(0, "gap")
    countdown(reader, "Debut dans", 3, on_window)
    for k, label in enumerate(order):
        name = names[label]
        hold(reader, cue, label, "cue", hold_s, f"[{k + 1:2d}/{len(order)}] >>> {CUE_TEXT.get(name, name)} <<<",
             on_window, sound="go" if label else "rest")
        cue.set(0, "gap")
        say("       relache")
        run_for(reader, gap_s, on_window)
    cue.set(0, "ready")
    beep("end")


def block_rows(log, block):
    return [row for row in log.rows if row["block"] == block]


def main():
    parser = argparse.ArgumentParser(description="Etape 5 : precision du decodeur en direct.")
    parser.add_argument("--cues", type=int, default=15, help="Nombre de consignes (reparties sur les 3 gestes)")
    parser.add_argument("--hold", type=float, default=4.0)
    parser.add_argument("--gap", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--model", type=str, default=None, help="Modele .json (defaut : le plus recent)")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    model_path = args.model or str(find_last_model())
    model = load_model(model_path)
    decoder = OnlineDecoder(model, vote_window=config.VOTE_WINDOW, adapt=config.ADAPT)
    seed = args.seed if args.seed is not None else random.randrange(10 ** 6)
    order = cue_order(args.cues, len(model.gestures), random.Random(seed))

    run_dir = new_run_dir("accuracy")
    cue, log = Cue(), DecoderLog()
    reader = None
    rest_measured = None
    try:
        with Sensor(cue, device_name=args.device) as reader:
            run_for(reader, 0.5)                    # transitoire de demarrage du filtre
            rest_measured = calibrate(reader, decoder, cue)
            run_test(reader, decoder, cue, log, order, args.hold, args.gap)
            say("\nTermine, merci.")
    finally:
        if reader is not None:
            reader.save(run_dir)
            log.save(run_dir / "decoder.csv")
            say(f"Enregistre : {run_dir} ({reader.lost} echantillons perdus)")

    result = live_metrics.analyse(log.rows, model.gestures)
    result.update({"model": model_path, "seed": seed, "order": [model.gestures[k] for k in order],
                   "hold_s": args.hold, "gap_s": args.gap, "rest_calibration_uV": rest_measured,
                   "vote_window": config.VOTE_WINDOW, "adapt": config.ADAPT, "lost_samples": reader.lost})
    result["targets"] = live_metrics.report(result, say)
    save_json(run_dir / "summary.json", result)
    say(f"Resultats : {run_dir}")


if __name__ == "__main__":
    main()
