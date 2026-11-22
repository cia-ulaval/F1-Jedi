"""Pont EMG -> manette virtuelle (vgamepad) pour piloter le jeu avec 3 gestes.

    python collect/game.py
    python collect/game.py --model_path models/f1jedi_3g_S3_....json

Touches (clavier global) :
    O  mode conduite (les gestes pilotent)      P  mode clavier (test manuel)
    R  recalibration rapide (~15 s)             Q  quitter

Mode conduite : accelerateur (bouton A) maintenu, les gestes dirigent selon
config.GAME_ACTIONS (par defaut repos = tout droit, Hand_Close = gauche,
Wrist_Extension = droite).
"""

import argparse
import time

try:
    from config import Config
    from emg_core import OnlineDecoder
    from realtime_test_etienne import calibrate_rest, guided_recalibration, load_model, open_sensor
except ImportError:
    from code_etienne.config import Config
    from code_etienne.emg_core import OnlineDecoder
    from code_etienne.realtime_test_etienne import calibrate_rest, guided_recalibration, load_model, open_sensor

config = Config()
MODEL_PATH = None
VALID_ACTIONS = ("straight", "steer_left", "steer_right", "brake")

try:
    import vgamepad as vg
except ImportError:
    vg = None

try:
    import keyboard
except ImportError:
    keyboard = None


def action_for_label(model, label):
    """Action de jeu associee a une classe predite (0 = repos, 1 = geste A, 2 = geste B)."""
    action = config.GAME_ACTIONS.get(model.gestures[label], "straight")
    if action not in VALID_ACTIONS:
        raise ValueError(f"Action inconnue '{action}' dans config.GAME_ACTIONS (attendu : {VALID_ACTIONS}).")
    return action


def apply_drive_action(gamepad, action):
    """Applique une action sur la manette (appelee apres gamepad.reset())."""
    if action == "brake":
        gamepad.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_B)
        return action
    if config.ACCELERATE_IN_DRIVE:
        gamepad.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_A)
    if action == "steer_left":
        gamepad.left_joystick(x_value=-config.STEER_VALUE, y_value=0)
    elif action == "steer_right":
        gamepad.left_joystick(x_value=config.STEER_VALUE, y_value=0)
    return action


def apply_keyboard_action(gamepad):
    action_names = []
    if keyboard.is_pressed("v"):
        gamepad.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_A)
        action_names.append("button_a")
    if keyboard.is_pressed("b"):
        gamepad.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_B)
        action_names.append("button_b")
    if keyboard.is_pressed("g"):
        gamepad.left_joystick(x_value=-config.STEER_VALUE, y_value=0)
        action_names.append("steer_left")
    if keyboard.is_pressed("j"):
        gamepad.left_joystick(x_value=config.STEER_VALUE, y_value=0)
        action_names.append("steer_right")
    if keyboard.is_pressed("y"):
        gamepad.left_joystick(x_value=0, y_value=config.STEER_VALUE)
        action_names.append("up")
    if keyboard.is_pressed("h"):
        gamepad.left_joystick(x_value=0, y_value=-config.STEER_VALUE)
        action_names.append("down")
    if keyboard.is_pressed("space"):
        gamepad.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_X)
        action_names.append("button_x")
    return ",".join(action_names) if action_names else "neutral"


def release_gamepad(gamepad):
    gamepad.reset()
    gamepad.update()


def run_realtime_game(model_path=None, vote_window=None, adapt=None, reader=None):
    if vg is None:
        raise ImportError("vgamepad n'est pas installe : pip install -r requirements.txt (Windows + pilote ViGEmBus).")
    if keyboard is None:
        raise ImportError("keyboard n'est pas installe : pip install -r requirements.txt")

    model = load_model(model_path)
    vote_window = config.VOTE_WINDOW if vote_window is None else vote_window
    adapt = config.ADAPT if adapt is None else adapt
    decoder = OnlineDecoder(model, vote_window=vote_window, adapt=adapt)
    for label, name in enumerate(model.gestures):
        print(f"[bridge] {name:<16} -> {action_for_label(model, label)}")

    gamepad = vg.VX360Gamepad()
    release_gamepad(gamepad)
    print("[bridge] Manette virtuelle creee")

    if reader is None:
        reader = open_sensor()
    calibrate_rest(reader, decoder)

    print("[bridge] Touches : O = conduite, P = clavier, R = recalibrer, Q = quitter")
    mode, last_report = "keyboard", None
    label, last_window_time = 0, time.perf_counter()
    try:
        while not getattr(reader, "finished", False):
            if keyboard.is_pressed("q"):
                print("[bridge] Arret demande")
                break
            if keyboard.is_pressed("o"):
                mode = "drive"
            elif keyboard.is_pressed("p"):
                mode = "keyboard"
            if keyboard.is_pressed("r"):
                # Manette relachee pendant toute la recalibration.
                release_gamepad(gamepad)
                guided_recalibration(reader, decoder)
                label, last_window_time = 0, time.perf_counter()
                continue

            window = reader.poll()
            now = time.perf_counter()
            if window is not None:
                label = decoder.update(window)["label"]
                last_window_time = now
            elif now - last_window_time > config.STALL_TIMEOUT:
                # Plus de donnees (Bluetooth coupe ?) : on ne garde pas un
                # virage enfonce a l'aveugle.
                label = 0

            gamepad.reset()
            if mode == "drive":
                stalled = now - last_window_time > config.STALL_TIMEOUT
                action = "no_signal" if stalled else apply_drive_action(gamepad, action_for_label(model, label))
            else:
                action = apply_keyboard_action(gamepad)
            gamepad.update()

            report = (mode, model.gestures[label], action)
            if report != last_report:
                print(f"[bridge] mode={mode} geste={model.gestures[label]} action={action}", flush=True)
                last_report = report
            time.sleep(config.DELAY_BETWEEN_PREDICTIONS)
    except KeyboardInterrupt:
        print("[bridge] Interrompu")
    finally:
        release_gamepad(gamepad)
        print("[bridge] Manette relachee")


def main():
    parser = argparse.ArgumentParser(description="Pilote une manette virtuelle avec 3 gestes EMG.")
    parser.add_argument("--model_path", type=str, default=MODEL_PATH, help="Modele .json (defaut : le plus recent)")
    parser.add_argument("--vote-window", "--majority-vote", type=int, default=config.VOTE_WINDOW,
                        help="Taille du vote majoritaire en fenetres (1 = desactive)")
    parser.add_argument("--no-adapt", action="store_true", help="Desactive le suivi de derive")
    args = parser.parse_args()
    run_realtime_game(args.model_path, vote_window=args.vote_window, adapt=not args.no_adapt)


if __name__ == "__main__":
    main()
