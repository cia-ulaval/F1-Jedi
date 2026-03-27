import argparse
import time
import csv
import re
from pathlib import Path

import numpy as np
import dearpygui.dearpygui as dpg
from PIL import Image  # pip install pillow

# =========================
# CONFIG
# =========================

POSES = [
    "Hand_Open",
    "Hand_Close",
    "No_Motion",
    "Wrist_Extension",
    "Wrist_Flexion",
]

POSE_TIME_S = 5.0
REST_TIME_S = 3.0

DATA_ROOT = Path("data")

WINDOW_W = 1100
WINDOW_H = 700
IMAGE_W = 420
IMAGE_H = 420


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def format_subject_id(raw: str) -> str:
    s = raw.strip().upper()
    s = s[1:] if s.startswith("S") else s
    if not s.isdigit():
        raise ValueError("Subject doit être un nombre (ex: 1, 02, 12).")
    return f"S{int(s):02d}"

def next_subject_id(data_root: Path = DATA_ROOT, prefix: str = "S", width: int = 2, start: int = 1) -> str:
    """
    Retourne le prochain ID libre: S01, S02, ...
    """
    data_root.mkdir(parents=True, exist_ok=True)
    existing = set()
    for p in data_root.iterdir():
        if p.is_dir() and p.name.startswith(prefix):
            suffix = p.name[len(prefix):]
            if suffix.isdigit():
                existing.add(int(suffix))

    n = start
    while n in existing:
        n += 1
    return f"{prefix}{n:0{width}d}"

def load_texture_from_png(path: Path):
    # IMPORTANT: on lit l'image, on ne la copie pas, on ne l'écrit pas.
    img = Image.open(path).convert("RGBA")
    w, h = img.size
    arr = np.asarray(img, dtype=np.float32) / 255.0
    flat = arr.reshape(-1).tolist()
    return w, h, flat


def write_csv(path: Path, array2d: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerows(array2d)


def wait_for_emg(odh, timeout_s: float = 8.0, modality: str = "emg"):
    """
    Attend jusqu'à ce qu'on reçoive au moins 1 échantillon EMG.
    """
    t0 = time.perf_counter()
    # reset pour partir propre
    try:
        odh.reset()
    except Exception:
        pass

    while time.perf_counter() - t0 < timeout_s:
        vals, count = odh.get_data()
        try:
            total = int(np.asarray(count[modality]).reshape(-1)[0])
        except Exception:
            total = 0
        if total > 0:
            return True
        time.sleep(0.05)
    return False


def patch_libemg_sifi_disconnect(libemg):
    """
    Compatibilite entre versions de sifi_bridge_py/libemg:
    disconnect() peut renvoyer soit un bool, soit un dict {"connected": bool}.
    """
    try:
        cls = libemg.streamers.SiFiBridgeStreamer
    except Exception:
        return

    if getattr(cls, "_disconnect_patched_for_bool", False):
        return

    def disconnect_compat(self):
        result = self.sb.disconnect()
        if isinstance(result, dict):
            self.connected = result.get("connected", False)
        else:
            self.connected = bool(result)
        return self.connected

    cls.disconnect = disconnect_compat
    cls._disconnect_patched_for_bool = True


def stop_streamer(streamer):
    try:
        if hasattr(streamer, "signal"):
            streamer.signal.set()
            streamer.join(timeout=5.0)
    except Exception:
        pass


def start_emg_session(timeout_s: float = 10.0):
    import libemg

    patch_libemg_sifi_disconnect(libemg)

    attempts = [
        (
            "BioPoint",
            lambda: libemg.streamers.sifi_biopoint_streamer(
                emg=True, imu=False, ppg=False, ecg=False, eda=False, streaming=False
            ),
        ),
        (
            "BioArmband",
            lambda: libemg.streamers.sifi_bioarmband_streamer(
                emg=True, imu=False, ppg=False, ecg=False, eda=False, streaming=False
            ),
        ),
    ]

    failures = []
    for device_label, factory in attempts:
        streamer = None
        keep_streamer = False
        try:
            streamer, smis = factory()
            odh = libemg.data_handler.OnlineDataHandler(shared_memory_items=smis)
            if hasattr(odh, "prepare_smm"):
                odh.prepare_smm()
            if wait_for_emg(odh, timeout_s=timeout_s, modality="emg"):
                keep_streamer = True
                print("=== DEVICE CONNECTED ===")
                print(device_label)
                return streamer, odh, "emg"
            failures.append(f"{device_label}: aucune donnee EMG recue avant {timeout_s:.0f}s")
        except Exception as exc:
            failures.append(f"{device_label}: {exc}")
        finally:
            if streamer is not None and not keep_streamer:
                stop_streamer(streamer)

    raise RuntimeError(
        "Impossible de recevoir des donnees EMG depuis SiFi Bridge. "
        + " | ".join(failures)
    )


def resolve_subject_id(subject_raw: str | None) -> str:
    if subject_raw is None or not subject_raw.strip():
        return next_subject_id()
    return format_subject_id(subject_raw)


def increment_subject_id(subject_id: str, step: int = 1) -> str:
    match = re.fullmatch(r"S(\d+)", subject_id)
    if not match:
        raise ValueError(f"Identifiant sujet invalide: {subject_id}")
    return f"S{int(match.group(1)) + step:02d}"


def collect_one_subject_all_poses(subject_raw: str | None = None, odh=None, modality: str = "emg"):
    """
    - 1 fenêtre
    - rest 3s -> pose 5s -> rest 3s -> ...
    - 1 seul fichier par pose: data/Sxx/<Pose>.csv (EMG only)
    - aucune copie/réécriture d'images
    """
    subject_id = resolve_subject_id(subject_raw)
    subject_folder = DATA_ROOT / subject_id
    if subject_folder.exists():
        raise FileExistsError(
            f"Le dossier {subject_folder} existe deja. Choisis un autre sujet ou supprime-le avant une nouvelle collecte."
        )
    subject_folder.mkdir(parents=True, exist_ok=True)

    # Images: directement depuis ./poses (à côté du script)
    project_root = Path(__file__).resolve().parent
    poses_folder = project_root / "poses"

    if not poses_folder.exists():
        raise RuntimeError(f"Dossier introuvable: {poses_folder.resolve()}")

    # Playlist strictement dans l'ordre POSES, en trouvant pose_name.png (case-insensitive)
    missing = []
    playlist = []
    for pose_name in POSES:
        found = None
        for cand in poses_folder.iterdir():
            if cand.is_file() and cand.suffix.lower() == ".png" and _norm(cand.stem) == _norm(pose_name):
                found = cand
                break
        if found is None:
            missing.append(pose_name)
        else:
            playlist.append((pose_name, found))

    if missing:
        raise RuntimeError("Images manquantes dans ./poses: " + ", ".join(missing))

    print("=== OUTPUT ROOT (requis) ===")
    print(DATA_ROOT.resolve())
    print("=== SUBJECT FOLDER ===")
    print(subject_folder.resolve())

    if odh is None:
        raise RuntimeError("Session EMG non initialisee.")

    ok = wait_for_emg(odh, timeout_s=10.0, modality=modality)
    if not ok:
        raise RuntimeError(
            "Aucune donnée EMG reçue après 10s. "
            "Vérifie SiFi Bridge / connexion bracelet / permissions."
        )

    # UI
    dpg.create_context()

    textures = {}
    with dpg.texture_registry(show=False):
        for pose_name, path in playlist:
            w, h, flat = load_texture_from_png(path)
            tag = f"tex_{pose_name}"
            dpg.add_static_texture(w, h, flat, tag=tag)
            textures[pose_name] = tag

    state = {
        "idx": 0,              # pose index
        "phase": "rest",       # rest / active
        "t0": time.perf_counter(),
        "duration": REST_TIME_S,
        "chunks": [],
        "count_seen": 0,
    }

    def cur_pose():
        return playlist[state["idx"]][0]

    def reset_buffers():
        odh.reset()
        state["chunks"] = []
        state["count_seen"] = 0

    def grab_new_data():
        vals, count = odh.get_data()
        total = int(np.asarray(count[modality]).reshape(-1)[0])
        new = total - state["count_seen"]
        if new > 0:
            chunk = np.asarray(vals[modality])[:new, :]
            state["chunks"].append(chunk)
            state["count_seen"] += new

    def save_pose():
        pose = cur_pose()
        if not state["chunks"]:
            raise RuntimeError(
                f"0 échantillon EMG reçu pendant {pose}. "
                "Le stream ne renvoie rien (SiFi Bridge ? device ?)."
            )
        data = np.vstack(state["chunks"])
        out = subject_folder / f"{pose}.csv"   # <-- UN SEUL FICHIER, PAS DE _emg
        write_csv(out, data)

    def set_ui():
        pose = cur_pose()
        if state["phase"] == "rest":
            dpg.set_value("title", f"{subject_id} — Pause")
            dpg.set_value("subtitle", f"Prochaine pose: {pose}")
            dpg.configure_item("img", texture_tag=textures[pose], tint_color=(150, 150, 150, 255))
        else:
            dpg.set_value("title", f"{subject_id} — {pose}")
            dpg.set_value("subtitle", "Tiens la pose jusqu'à la fin du timer")
            dpg.configure_item("img", texture_tag=textures[pose], tint_color=(255, 255, 255, 255))

    dpg.create_viewport(title="EMG Data Collection", width=WINDOW_W, height=WINDOW_H)
    dpg.setup_dearpygui()

    with dpg.window(tag="main", no_title_bar=True, width=WINDOW_W, height=WINDOW_H):
        dpg.add_spacer(height=16)
        dpg.add_text("", tag="title")
        dpg.add_text("", tag="subtitle")
        dpg.add_spacer(height=12)

        with dpg.group(horizontal=True):
            dpg.add_spacer(width=(WINDOW_W - IMAGE_W) // 2 - 10)
            dpg.add_image(textures[cur_pose()], tag="img", width=IMAGE_W, height=IMAGE_H)

        dpg.add_spacer(height=16)
        dpg.add_progress_bar(tag="prog", default_value=0.0, width=IMAGE_W + 120)
        dpg.add_spacer(height=10)
        dpg.add_text("ESC pour quitter", color=(170, 170, 170))

    # init
    reset_buffers()
    set_ui()
    dpg.show_viewport()

    while dpg.is_dearpygui_running():
        if dpg.is_key_down(dpg.mvKey_Escape):
            dpg.stop_dearpygui()

        elapsed = time.perf_counter() - state["t0"]
        dpg.set_value("prog", min(1.0, elapsed / max(1e-6, state["duration"])))

        if state["phase"] == "active":
            grab_new_data()

        if elapsed >= state["duration"]:
            if state["phase"] == "rest":
                state["phase"] = "active"
                state["t0"] = time.perf_counter()
                state["duration"] = POSE_TIME_S
                reset_buffers()
                set_ui()
            else:
                save_pose()

                state["idx"] += 1
                if state["idx"] >= len(playlist):
                    dpg.stop_dearpygui()
                    break

                state["phase"] = "rest"
                state["t0"] = time.perf_counter()
                state["duration"] = REST_TIME_S
                set_ui()

        dpg.render_dearpygui_frame()

    dpg.destroy_context()


def collect_repetitions(subject_raw: str | None = None, repetitions: int = 1):
    if repetitions < 1:
        raise ValueError("--repetitions doit etre >= 1.")

    current_subject = resolve_subject_id(subject_raw) if subject_raw else None
    streamer, odh, modality = start_emg_session(timeout_s=10.0)

    try:
        for rep_idx in range(repetitions):
            if current_subject is None:
                subject_for_run = None
            else:
                subject_for_run = current_subject

            print(f"=== REPETITION {rep_idx + 1}/{repetitions} ===")
            collect_one_subject_all_poses(subject_for_run, odh=odh, modality=modality)

            if current_subject is not None:
                current_subject = increment_subject_id(current_subject)
    finally:
        stop_streamer(streamer)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collecte EMG pour un sujet et les 5 poses.")
    parser.add_argument(
        "--subject",
        type=str,
        default=None,
        help="Identifiant sujet, ex: 2 ou S02. Si absent, le prochain ID libre est utilise.",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=1,
        help="Nombre de collectes consecutives a executer. Chaque repetition est enregistree comme un sujet distinct.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    collect_repetitions(args.subject, args.repetitions)
