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


def collect_one_subject_all_poses(subject_raw: str):
    """
    - 1 fenêtre
    - rest 3s -> pose 5s -> rest 3s -> ...
    - 1 seul fichier par pose: data/Sxx/<Pose>.csv (EMG only)
    - aucune copie/réécriture d'images
    """
    subject_id = next_subject_id()
    subject_folder = DATA_ROOT / subject_id
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

    # Import libemg (ton env doit être ok)
    import libemg

    # EMG only
    streamer, smis = libemg.streamers.sifi_biopoint_streamer(
        emg=True, imu=False, ppg=False, ecg=False, eda=False, streaming=False
    )

    odh = libemg.data_handler.OnlineDataHandler(shared_memory_items=smis)
    modality = "emg"
    ok = wait_for_emg(odh, timeout_s=10.0, modality=modality)
    if not ok:
        raise RuntimeError(
            "Aucune donnée EMG reçue après 10s. "
            "Vérifie SiFi Bridge / connexion bracelet / permissions."
        )

    if hasattr(odh, "prepare_smm"):
        odh.prepare_smm()

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

    # stop streamer
    try:
        if hasattr(streamer, "signal"):
            streamer.signal.set()
    except Exception:
        pass

    dpg.destroy_context()


if __name__ == "__main__":
    # mets "1", "2", "12", etc.
    collect_one_subject_all_poses("1")