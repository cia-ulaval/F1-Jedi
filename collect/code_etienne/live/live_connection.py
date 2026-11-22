"""Etape 1 - connexion : lit le BioPoint N secondes avec le WindowReader du jeu.

    python collect/code_etienne/live/live_connection.py --duration 10
    python collect/code_etienne/live/live_connection.py --scan          # liste les capteurs BLE
    python collect/code_etienne/live/live_connection.py --device BioPoint_v1_3

Mesure : frequence d'echantillonnage reelle, regularite des paquets (trous),
ordre chronologique des fenetres (comparees au flux brut enregistre).
Resultats : dataset/live/<date>_connection/{emg.csv, summary.json}.
"""

import argparse
import subprocess
import sys
import time

import numpy as np

from emg_core import frame_signal
from live_common import NO_CUE, Cue, Sensor, config, new_run_dir, run_for, save_json, say


def scan(timeout_s=10.0):
    import sifi_bridge_py as sbp
    bridge = sbp.SifiBridge()
    try:
        return bridge.list_devices(sbp.ListSources.BLE)
    finally:
        bridge._bridge.kill()


def sifibridge_processes():
    if sys.platform != "win32":
        return []
    out = subprocess.run(["tasklist", "/FI", "IMAGENAME eq sifibridge.exe", "/FO", "CSV", "/NH"],
                         capture_output=True, text=True).stdout
    return [line for line in out.splitlines() if "sifibridge" in line.lower()]


def analyse(reader, windows):
    reads = np.array(reader.reads)                       # (t, total) a chaque lecture
    idx, t, emg, _, _ = reader.tape()
    result = {"duration_s": float(reads[-1, 0] - reads[0, 0]) if len(reads) else 0.0,
              "samples": int(len(emg)), "lost_samples": int(reader.lost)}

    # Frequence : instants ou le compteur change (= arrivee d'un paquet).
    change = np.flatnonzero(np.diff(reads[:, 1]) > 0) + 1
    arrivals = reads[change]
    if len(arrivals) >= 2:
        span_t = arrivals[-1, 0] - arrivals[0, 0]
        result["sample_rate_hz"] = float((arrivals[-1, 1] - arrivals[0, 1]) / span_t)
        gaps = np.diff(arrivals[:, 0])
        sizes = np.diff(arrivals[:, 1])
        result["packets"] = int(len(arrivals))
        result["packet_size_median"] = float(np.median(sizes))
        result["packet_size_values"] = {int(k): int(v) for k, v in zip(*np.unique(sizes, return_counts=True))}
        result["gap_median_ms"] = float(1e3 * np.median(gaps))
        result["gap_p99_ms"] = float(1e3 * np.percentile(gaps, 99))
        result["gap_max_ms"] = float(1e3 * gaps.max())
        result["gaps_over_100ms"] = int(np.sum(gaps > 0.1))
        # Debit par seconde (montre un eventuel trou d'une seconde).
        per_s = []
        for start in np.arange(arrivals[0, 0], arrivals[-1, 0] - 1.0, 1.0):
            inside = (arrivals[:, 0] >= start) & (arrivals[:, 0] < start + 1.0)
            per_s.append(int(np.sum(sizes[inside[1:]])))
        result["samples_per_second"] = per_s

    # Continuite du flux enregistre.
    result["tape_contiguous"] = bool(len(idx) == 0 or np.array_equal(idx, np.arange(idx[0], idx[0] + len(idx))))

    # Fenetres : identiques au flux brut, et dans l'ordre chronologique.
    ends = np.array([end for end, _ in windows])
    mismatches, worst_shift = 0, 0
    first = int(idx[0]) if len(idx) else 0
    for end, window in windows:
        start = end - config.WINDOW_SIZE - first
        expected = emg[start:start + config.WINDOW_SIZE] if start >= 0 else None
        if expected is None or len(expected) != config.WINDOW_SIZE or not np.array_equal(expected, window):
            mismatches += 1
            # Cherche le decalage qui explique la fenetre (course compteur/donnees).
            for shift in range(-200, 201):
                s = start + shift
                if 0 <= s and s + config.WINDOW_SIZE <= len(emg) and np.array_equal(emg[s:s + config.WINDOW_SIZE], window):
                    worst_shift = max(worst_shift, abs(shift), key=abs)
                    break
    steps = np.diff(ends) if len(ends) > 1 else np.array([])
    result["windows"] = int(len(windows))
    result["windows_per_s"] = float(len(windows) / max(result["duration_s"], 1e-9))
    result["windows_mismatch_vs_tape"] = int(mismatches)
    result["window_shift_max_samples"] = int(worst_shift)
    result["windows_chronological"] = bool(np.all(steps > 0)) if len(steps) else True
    result["window_step_values"] = {int(k): int(v) for k, v in zip(*np.unique(steps, return_counts=True))} if len(steps) else {}
    # Ordre interne : la fenetre doit etre le flux dans le sens du temps (pas inverse).
    if windows:
        last_end, last_window = windows[-1]
        start = last_end - config.WINDOW_SIZE - first
        result["last_window_equals_reversed_tape"] = bool(
            start >= 0 and np.array_equal(emg[start:start + config.WINDOW_SIZE][::-1], last_window))

    # RMS par fenetre de 200 ms. Le filtre du capteur met ~100 ms a se stabiliser
    # au demarrage (mesure : 437 uV sur la 1re fenetre, puis ~7 uV) : on mesure
    # le repos apres 0.5 s, et le transitoire a part.
    blocks = frame_signal(emg, config.WINDOW_SIZE, config.WINDOW_SIZE)
    if len(blocks):
        rms = np.sqrt(np.mean((blocks - blocks.mean(axis=1, keepdims=True)) ** 2, axis=1)) * 1e6
        settled = rms[int(0.5 * config.EMG_FS) // config.WINDOW_SIZE:]
        result["startup_transient_uV"] = float(rms[0])
        result["rest_rms_uV_median"] = float(np.median(settled)) if len(settled) else None
        result["rest_rms_uV_p90"] = float(np.percentile(settled, 90)) if len(settled) else None
    return result


def main():
    parser = argparse.ArgumentParser(description="Etape 1 : connexion et frequence reelle du BioPoint.")
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--device", type=str, default=None, help=f"Nom du capteur (defaut {config.DEVICE_NAME})")
    parser.add_argument("--scan", action="store_true", help="Liste les capteurs BLE visibles et quitte")
    args = parser.parse_args()

    if args.scan:
        say(f"Capteurs BLE : {scan()}")
        return

    say(f"sifibridge avant : {len(sifibridge_processes())} processus")
    run_dir = new_run_dir("connection")
    cue = Cue()
    windows = []
    reader = None
    t_start = time.perf_counter()
    try:
        with Sensor(cue, device_name=args.device) as reader:
            cue.set(NO_CUE, "connection")
            say(f"Lecture pendant {args.duration:g} s (bras au repos, rien a faire)...")
            run_for(reader, args.duration, lambda t, w: windows.append((reader.last_end, w.copy())))
    finally:
        if reader is not None:
            reader.save(run_dir)
    time.sleep(1.0)
    leftover = sifibridge_processes()

    result = analyse(reader, windows)
    result["device"] = args.device or config.DEVICE_NAME
    result["wall_s"] = time.perf_counter() - t_start
    result["sifibridge_left_after_stop"] = leftover
    save_json(run_dir / "summary.json", result)
    for key, value in result.items():
        if key not in ("samples_per_second", "packet_size_values"):
            say(f"  {key:<34} {value}")
    say(f"  samples_per_second                 {result.get('samples_per_second')}")
    say(f"  packet_size_values                 {result.get('packet_size_values')}")
    say(f"Resultats : {run_dir}")


if __name__ == "__main__":
    main()
