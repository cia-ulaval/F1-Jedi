"""Rejoue hors ligne un enregistrement dataset/live/<run> dans le decodeur.

    python collect/code_etienne/live/live_replay.py dataset/live/<run>
    python collect/code_etienne/live/live_replay.py dataset/live/<run> --vote-window 5 --rest-tau 10

Les fenetres sont recoupees dans le flux brut exactement comme en direct
(pas de 40 echantillons depuis le debut). Conventions de phase :
  calib            -> calibrate_rest (fenetres non decodees)
  recal_0/1/2      -> recalibrate(repos, geste A, geste B) a la fin de la sequence
  calib_settle, recal_prep -> ignorees (ni decodees ni utilisees)
  le reste         -> decoder.update, comme en jeu
Sans option, le resultat doit reproduire celui du direct (meme modele, memes reglages).
"""

import argparse
import json
from pathlib import Path

import numpy as np

from live_common import config, load_tape, say
import live_metrics
from emg_core import OnlineDecoder, ThreeGestureModel

IGNORED = {"calib_settle", "recal_prep", "setup"}


def replay_rows(tape, decoder, verbose=False, exact=True):
    ws, inc = config.WINDOW_SIZE, config.WINDOW_INC
    emg, phase, cue, block, t = tape["emg"], tape["phase"], tape["cue"], tape["block"], tape["t"]
    # Debut de chaque plage (cue, phase, bloc) -> since_cue.
    key_change = np.ones(len(emg), dtype=bool)
    key_change[1:] = (cue[1:] != cue[:-1]) | (phase[1:] != phase[:-1]) | (block[1:] != block[:-1])
    onset_index = np.maximum.accumulate(np.where(key_change, np.arange(len(emg)), 0))

    # Fenetres : celles rendues en direct (windows.txt) si disponibles -> replay
    # exact meme apres un saut ; sinon la grille reguliere (jeu sans a-coups).
    first = int(tape["sample"][0]) if len(tape["sample"]) else 0
    live_ends = tape.get("window_ends")
    if exact and live_ends is not None and len(live_ends):
        ends = [int(e) - first for e in live_ends if ws <= int(e) - first <= len(emg)]
    else:
        ends = range(ws, len(emg) + 1, inc)
    rows, calib, recal = [], [], {}
    for end in ends:
        k = end - 1
        window = emg[end - ws:end]
        ph = str(phase[k])
        if recal and not ph.startswith("recal_"):
            _apply_recal(decoder, recal, verbose)
            recal = {}
        if ph != "calib" and calib:
            measured = decoder.calibrate_rest(np.array(calib))
            if verbose:
                say(f"  calibration du repos : {np.exp(measured) * 1e6:.1f} uV")
            calib = []
        if ph in IGNORED:
            continue
        if ph == "calib":
            calib.append(window)
            continue
        if ph.startswith("recal_"):
            recal.setdefault(int(ph.split("_")[1]), []).append(window)
            continue
        out = decoder.update(window)
        rows.append({"t_s": float(t[k]), "end_sample": int(end + first), "block": int(block[k]), "cue": int(cue[k]), "phase": ph,
                     "since_cue_s": float(t[k] - t[onset_index[k]]), "label": out["label"],
                     "raw_label": out["raw_label"], "rms_uV": out["rms_uV"],
                     "threshold_uV": out["threshold_uV"], "rest_uV": out["rest_uV"]})
    return rows


def _apply_recal(decoder, recal, verbose):
    try:
        levels = decoder.recalibrate(*(np.array(recal.get(k, [])) for k in (0, 1, 2)))
        if verbose:
            say(f"  recalibration : repos {levels['rest_uV']:.1f} uV, actif {levels['active_uV']:.1f} uV")
    except ValueError as error:
        say(f"  recalibration refusee : {error}")


def build_decoder(model, args):
    kwargs = {}
    for name in ("rest_tau_s", "active_tau_s", "shift_tau_s"):
        value = getattr(args, name, None)
        if value is not None:
            kwargs[name] = value
    return OnlineDecoder(model, vote_window=args.vote_window, adapt=not args.no_adapt, **kwargs)


def main():
    parser = argparse.ArgumentParser(description="Rejoue un enregistrement live dans le decodeur.")
    parser.add_argument("run_dir")
    parser.add_argument("--model", type=str, default=None, help="Defaut : celui utilise en direct (summary.json)")
    parser.add_argument("--vote-window", type=int, default=config.VOTE_WINDOW)
    parser.add_argument("--no-adapt", action="store_true")
    parser.add_argument("--rest-tau", dest="rest_tau_s", type=float, default=None)
    parser.add_argument("--active-tau", dest="active_tau_s", type=float, default=None)
    parser.add_argument("--shift-tau", dest="shift_tau_s", type=float, default=None)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--regular-grid", action="store_true",
                        help="Fenetres tous les 40 echantillons au lieu de celles rendues en direct")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    summary_path = run_dir / "summary.json"
    live = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    model_path = args.model or live.get("model")
    if not model_path:
        from realtime_test_etienne import find_last_model
        model_path = str(find_last_model())
    model = ThreeGestureModel.load(model_path)
    say(f"Replay de {run_dir.name} avec {Path(model_path).name}")
    rows = replay_rows(load_tape(run_dir), build_decoder(model, args), verbose=not args.quiet,
                       exact=not args.regular_grid)
    for block in sorted({row["block"] for row in rows}):
        result = live_metrics.analyse([r for r in rows if r["block"] == block], model.gestures)
        say(f"\n--- bloc {block}")
        live_metrics.report(result, say)
    return rows


if __name__ == "__main__":
    main()
