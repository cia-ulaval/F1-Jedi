"""Metriques du test de precision (etapes 5 et 6), communes au direct et au replay.

Une ligne par fenetre decodee : t_s, cue, phase, since_cue_s, label, raw_label,
rms_uV, threshold_uV, rest_uV. Phases : 'cue' (geste demande), 'gap' (repos
entre deux consignes, verite = No_Motion). Les autres phases sont ignorees.
"""

import numpy as np

SKIP_S = 0.4


def _window_inc():
    from live_common import config
    return config.WINDOW_INC


def periods(rows):
    """Plages contigues de (phase, cue) -> liste de dicts avec leurs lignes."""
    out = []
    for row in rows:
        key = (row["phase"], row["cue"])
        if not out or out[-1]["key"] != key:
            out.append({"key": key, "phase": row["phase"], "cue": row["cue"], "rows": []})
        out[-1]["rows"].append(row)
    return out


def truth_of(period):
    return period["cue"] if period["phase"] == "cue" else 0


def analyse(rows, gestures, skip_s=SKIP_S):
    blocks = [p for p in periods(rows) if p["phase"] in ("cue", "gap") and p["cue"] >= 0]
    confusion = np.zeros((3, 3), dtype=int)
    onset_latency = {1: [], 2: []}      # consigne -> 1re sortie correcte
    decoder_latency = {1: [], 2: []}    # 1re fenetre au-dessus du seuil -> 1re sortie correcte
    release_latency = []                # retour au repos -> sortie No_Motion
    missed = {0: 0, 1: 0, 2: 0}
    flicker_cue, flicker_rest, cue_count = 0, 0, {0: 0, 1: 0, 2: 0}
    previous = None
    for block in blocks:
        truth = truth_of(block)
        t0 = block["rows"][0]["t_s"] - block["rows"][0]["since_cue_s"]
        scored = [r for r in block["rows"] if r["since_cue_s"] >= skip_s]
        for row in scored:
            confusion[truth, row["label"]] += 1
        labels = [r["label"] for r in scored]
        switches = sum(1 for a, b in zip(labels, labels[1:]) if a != b)
        if block["phase"] == "cue":
            cue_count[truth] += 1
            flicker_cue += switches
            if truth in onset_latency:
                hit = next((r for r in block["rows"] if r["label"] == truth), None)
                if hit is None:
                    missed[truth] += 1
                else:
                    onset_latency[truth].append(hit["t_s"] - t0)
                    active = next((r for r in block["rows"] if r["raw_label"] != 0), None)
                    if active is not None and active["t_s"] <= hit["t_s"]:
                        decoder_latency[truth].append(hit["t_s"] - active["t_s"])
        else:
            flicker_rest += switches
            if previous is not None and previous["phase"] == "cue" and previous["cue"] in (1, 2):
                hit = next((r for r in block["rows"] if r["label"] == 0), None)
                if hit is not None:
                    release_latency.append(hit["t_s"] - t0)
        previous = block

    total = max(int(confusion.sum()), 1)
    all_onsets = onset_latency[1] + onset_latency[2]
    all_decoder = decoder_latency[1] + decoder_latency[2]

    def med_ms(values):
        return float(1e3 * np.median(values)) if values else None

    last = rows[-1] if rows else {}
    ends = np.array([r.get("end_sample", -1) for r in rows])
    steps = np.diff(ends[ends >= 0])
    return {
        "skipped_window_jumps": int(np.sum(steps != _window_inc())) if steps.size else None,
        "accuracy": float(np.trace(confusion) / total),
        "accuracy_cues_only": _accuracy_where(rows, blocks, skip_s, phase="cue"),
        "confusion": confusion.tolist(),
        "gestures": list(gestures),
        "cues": {gestures[k]: v for k, v in cue_count.items()},
        "latency_median_ms": med_ms(all_onsets),
        "latency_median_ms_per_gesture": {gestures[k]: med_ms(v) for k, v in onset_latency.items()},
        "latency_max_ms": float(1e3 * max(all_onsets)) if all_onsets else None,
        "decoder_latency_median_ms": med_ms(all_decoder),
        "release_latency_median_ms": med_ms(release_latency),
        "missed_cues": {gestures[k]: v for k, v in missed.items() if k in (1, 2)},
        "flicker_during_cues": int(flicker_cue),
        "flicker_during_rest_gaps": int(flicker_rest),
        "end_rest_uV": last.get("rest_uV"),
        "end_threshold_uV": last.get("threshold_uV"),
    }


def _accuracy_where(rows, blocks, skip_s, phase):
    hits = total = 0
    for block in blocks:
        if block["phase"] != phase:
            continue
        truth = truth_of(block)
        for row in block["rows"]:
            if row["since_cue_s"] >= skip_s:
                hits += row["label"] == truth
                total += 1
    return float(hits / total) if total else None


def confusion_text(confusion, gestures):
    confusion = np.asarray(confusion)
    width = max(len(name) for name in gestures) + 2
    lines = [" " * width + "".join(f"{name:>{width}}" for name in gestures) + "   <- predit"]
    for row, name in enumerate(gestures):
        total = max(confusion[row].sum(), 1)
        lines.append(f"{name:<{width}}" + "".join(f"{confusion[row, col] / total:>{width}.1%}" for col in range(3)))
    return "\n".join(lines)


def report(result, say=print):
    fmt = lambda v: "n/a" if v is None else f"{v:.0f} ms"
    say("\n=== Precision en direct ===")
    say(f"  Exactitude globale (sans les {SKIP_S * 1e3:.0f} ms apres chaque consigne) : {result['accuracy']:.1%}"
        f"   [consignes seules : {result['accuracy_cues_only']:.1%}]")
    say(confusion_text(result["confusion"], result["gestures"]))
    per = ", ".join(f"{k} {fmt(v)}" for k, v in result["latency_median_ms_per_gesture"].items())
    say(f"  Latence consigne -> bonne sortie : mediane {fmt(result['latency_median_ms'])} ({per}), "
        f"max {fmt(result['latency_max_ms'])}")
    say(f"    dont decodeur seul (activation -> bonne sortie) : mediane {fmt(result['decoder_latency_median_ms'])}")
    say(f"  Retour au repos -> No_Motion : mediane {fmt(result['release_latency_median_ms'])}")
    say(f"  Consignes jamais reconnues : {result['missed_cues']}")
    say(f"  Clignotement : {result['flicker_during_cues']} changements pendant les gestes tenus, "
        f"{result['flicker_during_rest_gaps']} pendant les repos")
    if result.get("skipped_window_jumps"):
        say(f"  ATTENTION : {result['skipped_window_jumps']} sauts de fenetres (lecture en retard sur le capteur)")
    say(f"  Etat du decodeur a la fin : repos {result['end_rest_uV']:.1f} uV, seuil {result['end_threshold_uV']:.1f} uV")
    ok = {"accuracy_90pct": result["accuracy"] >= 0.90,
          "latency_below_400ms": result["latency_median_ms"] is not None and result["latency_median_ms"] < 400,
          "flicker_near_0": result["flicker_during_cues"] <= 2}
    say(f"  Cibles : {ok}")
    return ok
