"""Aide au placement du capteur : RMS en direct et rapport par rapport au repos.

    python collect/code_etienne/live/live_rms_meter.py --duration 90

Les 3 premieres secondes mesurent le repos (main detendue). Ensuite, 4 fois par
seconde : RMS (uV), rapport / repos, et une barre. Fais Wrist_Extension puis
Hand_Close, detends, deplace le capteur, recommence. Cible : >= 5x pour les deux.
Le signal est enregistre dans dataset/live/<date>_meter/emg.csv.
"""

import argparse
import time

import numpy as np

from live_common import NO_CUE, Cue, Sensor, config, new_run_dir, run_for, say

REFRESH_S = 0.25


def bar(ratio, width=40, full_scale=10.0):
    filled = int(min(ratio / full_scale, 1.0) * width)
    target = int(5.0 / full_scale * width)
    cells = ["#" if k < filled else ("|" if k == target else ".") for k in range(width)]
    return "".join(cells)


def main():
    parser = argparse.ArgumentParser(description="RMS en direct pour placer le capteur.")
    parser.add_argument("--duration", type=float, default=90.0)
    parser.add_argument("--rest", type=float, default=3.0, help="Duree de la mesure du repos au debut (s)")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    run_dir = new_run_dir("meter")
    cue = Cue()
    reader = None
    try:
        with Sensor(cue, device_name=args.device) as reader:
            run_for(reader, 0.5)                     # transitoire de demarrage du filtre
            say(f"Repos : main detendue {args.rest:g} s...")
            rest_windows = []
            run_for(reader, args.rest, lambda t, w: rest_windows.append(w.std()))
            rest = float(np.median(rest_windows)) if rest_windows else 1e-6
            say(f"Repos = {rest * 1e6:.1f} uV. Barre : 1 '#' = 0.25x, '|' = cible 5x. "
                f"Fais tes gestes ({args.duration:g} s).\n")
            cue.set(NO_CUE, "meter")
            recent = []
            next_print = time.perf_counter() + REFRESH_S
            deadline = time.perf_counter() + args.duration
            peak = {"t": 0.0, "ratio": 0.0}
            while time.perf_counter() < deadline:
                window = reader.poll()
                if window is None:
                    time.sleep(config.DELAY_BETWEEN_PREDICTIONS)
                else:
                    recent.append(window[-int(REFRESH_S * config.EMG_FS):].std())
                now = time.perf_counter()
                if now >= next_print and recent:
                    rms = float(np.median(recent))
                    ratio = rms / rest
                    if ratio > peak["ratio"] or now - peak["t"] > 2.0:
                        peak = {"t": now, "ratio": ratio}
                    left = deadline - now
                    print(f"{rms * 1e6:7.1f} uV  {ratio:5.1f}x  {bar(ratio)}  max 2 s {peak['ratio']:5.1f}x  "
                          f"[{left:3.0f} s]", flush=True)
                    recent, next_print = [], now + REFRESH_S
    finally:
        if reader is not None:
            reader.save(run_dir)
            say(f"Signal enregistre : {run_dir / 'emg.csv'}")


if __name__ == "__main__":
    main()
