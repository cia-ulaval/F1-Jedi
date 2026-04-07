import argparse
import time
from pathlib import Path

import numpy as np

from libemg.emg_predictor import OnlineEMGClassifier
from libemg.environments.controllers import ClassifierController

try:
    from config import Config
    from collect_etienne import prepare_streamer
    from realtime_test_etienne import install_online_filters, load_model, map_class_label, wait_for_emg
except ImportError:
    from code_etienne.config import Config
    from code_etienne.collect_etienne import prepare_streamer
    from code_etienne.realtime_test_etienne import install_online_filters, load_model, map_class_label, wait_for_emg


config = Config()
MODEL_PATH = None
GAME_ACTIONS = {
    1: ("accelerate", "right_trigger"),
    2: ("brake", "left_trigger"),
    3: ("neutral", "none"),
    5: ("steer_left", "left_joystick_negative"),
    4: ("steer_right", "left_joystick_positive"),
}

try:
    import vgamepad as vg
except ImportError:
    vg = None

try:
    import keyboard
except ImportError:
    keyboard = None


def apply_action(gamepad, label):
    action_name, action_code = GAME_ACTIONS.get(label, ("neutral", "none"))

    if action_code == "right_trigger":
        gamepad.right_trigger(value=255)
    elif action_code == "left_trigger":
        gamepad.left_trigger(value=255)
    elif action_code == "left_joystick_negative":
        gamepad.left_joystick(x_value=-20000, y_value=0)
    elif action_code == "left_joystick_positive":
        gamepad.left_joystick(x_value=20000, y_value=0)

    return action_name


def apply_drive_action(gamepad, prediction_idx):
    # Keep acceleration held in drive mode.
    gamepad.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_A)

    if prediction_idx == 0:  # Hand_Open
        return "accelerate_only"
    if prediction_idx == 1:  # Hand_Close
        gamepad.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_B)
        return "button_b"
    if prediction_idx == 2:  # No_Motion
        return "neutral"
    if prediction_idx == 4:  # Wrist_Extension
        gamepad.left_joystick(x_value=20000, y_value=0)
        return "steer_right"
    if prediction_idx == 3:  # Wrist_Flexion
        gamepad.left_joystick(x_value=-20000, y_value=0)
        return "steer_left"
    return "unknown_prediction"


def apply_keyboard_action(gamepad):
    action_names = []

    if keyboard.is_pressed("z"):
        gamepad.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_A)
        action_names.append("button_a")
    if keyboard.is_pressed("x"):
        gamepad.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_B)
        action_names.append("button_b")
    if keyboard.is_pressed("a"):
        gamepad.left_joystick(x_value=-20000, y_value=0)
        action_names.append("steer_left")
    if keyboard.is_pressed("d"):
        gamepad.left_joystick(x_value=20000, y_value=0)
        action_names.append("steer_right")
    if keyboard.is_pressed("w"):
        gamepad.left_joystick(x_value=0, y_value=20000)
        action_names.append("up")
    if keyboard.is_pressed("s"):
        gamepad.left_joystick(x_value=0, y_value=-20000)
        action_names.append("down")
    if keyboard.is_pressed("space"):
        gamepad.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_X)
        action_names.append("button_x")

    return ",".join(action_names) if action_names else "neutral"


def resolve_model_path(model_path):
    bridge_dir = Path(__file__).resolve().parent
    repo_root = bridge_dir.parent.parent

    if model_path:
        candidate = Path(model_path)
        if not candidate.is_absolute():
            candidate = (Path.cwd() / candidate).resolve()
        return str(candidate)

    model_dir = repo_root / "models"
    subject = config.SESSION_TRAIN
    model_files = list(model_dir.glob(f"libemg_lda_S{subject}_*.pkl"))
    if not model_files:
        raise FileNotFoundError(f"No model files found for subject S{subject} in {model_dir}")

    latest_model = max(model_files, key=lambda f: f.stat().st_mtime)
    print(f"[bridge] Latest model found: {latest_model}")
    return str(latest_model)


def run_realtime_game(model_path, majority_vote=config.MAJORITY_VOTE_WINDOW, delay=config.DELAY_BETWEEN_PREDICTIONS, use_filter=config.USE_FILTERS, confidence_threshold=0.60):
    if vg is None:
        raise ImportError("vgamepad is not installed. Run '.\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt'.")
    if keyboard is None:
        raise ImportError("keyboard is not installed. Run '.\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt'.")

    if majority_vote is None:
        majority_vote = config.MAJORITY_VOTE_WINDOW
    if delay is None:
        delay = config.DELAY_BETWEEN_PREDICTIONS

    print("[bridge] Starting game bridge")
    print(f"[bridge] model_path={model_path or 'latest-for-session'} majority_vote={majority_vote} delay={delay} confidence_threshold={confidence_threshold} use_filter={use_filter}")

    window_size = int(config.WINDOW_SIZE_MS * config.EMG_FS / 1000)
    window_inc = int(config.WINDOW_INC_MS * config.EMG_FS / 1000)

    resolved_model_path = resolve_model_path(model_path)
    print(f"[bridge] Resolved model path={resolved_model_path}")
    print("[bridge] Loading model")
    model = load_model(resolved_model_path)
    if majority_vote and majority_vote > 1:
        model.add_majority_vote(num_samples=majority_vote)
        print(f"[bridge] Majority vote enabled with window={majority_vote}")
    else:
        print("[bridge] Majority vote disabled")

    print("[bridge] Preparing EMG streamer")
    odh = prepare_streamer(streaming=False)
    if use_filter:
        print("[bridge] Installing online filters")
        install_online_filters(odh)
    else:
        print("[bridge] Online filters disabled")

    print(f"[bridge] Waiting for EMG data for up to {config.TIMEOUT} seconds")
    if not wait_for_emg(odh, timeout_s=config.TIMEOUT):
        raise RuntimeError(f"No EMG samples received from device after {config.TIMEOUT} seconds.")
    print("[bridge] EMG data detected")

    features_list = ["LS", "MFL", "MSR", "WAMP", "WENG"]
    print(f"[bridge] Features={features_list}")

    num_classes = config.NUM_CLASSES
    try:
        num_classes = len(model.model.classes_)
    except Exception:
        pass
    print(f"[bridge] num_classes={num_classes}")

    oclassi = OnlineEMGClassifier(
        offline_classifier=model,
        window_size=window_size,
        window_increment=window_inc,
        online_data_handler=odh,
        features=features_list,
        std_out=False,
        output_format="probabilities",
    )
    controller = ClassifierController(
        output_format="probabilities",
        num_classes=num_classes,
    )
    gamepad = vg.VX360Gamepad()
    print("[bridge] Virtual gamepad created")
    mode = "keyboard"
    last_prediction = 2
    last_confidence = 0.0
    last_mode_printed = None

    try:
        odh.reset()
    except Exception:
        pass

    print("[bridge] Starting EMG classifier loop")
    print("[bridge] Press Ctrl+C to stop")
    print("[bridge] Controls: O=drive mode, P=keyboard mode, Q=quit")

    try:
        oclassi.run(block=False)
        print("[bridge] Classifier running in non-blocking mode")
        while True:
            gamepad.reset()

            if keyboard.is_pressed("q"):
                print("[bridge] Quit requested from keyboard")
                break
            if keyboard.is_pressed("o"):
                mode = "drive"
            elif keyboard.is_pressed("p"):
                mode = "keyboard"

            if mode != last_mode_printed:
                print(f"[bridge] Mode switched to {mode}")
                last_mode_printed = mode

            data = controller.get_data(["probabilities", "timestamp"])
            if data is not None:
                probabilities, timestamp = data
                probs = np.asarray(probabilities, dtype=float)
                pred_idx = int(np.argmax(probs))
                confidence = float(np.max(probs))
                label = map_class_label(pred_idx)
                if confidence >= confidence_threshold:
                    last_prediction = pred_idx
                    last_confidence = confidence

                if mode == "drive":
                    action_name = apply_drive_action(gamepad, last_prediction)
                elif mode == "keyboard":
                    action_name = apply_keyboard_action(gamepad)

                print(
                    f"[bridge] t={float(timestamp):.3f} mode={mode} pred_idx={pred_idx} mapped_label={label} active_prediction={last_prediction} action={action_name} confidence={confidence:.3f} last_confidence={last_confidence:.3f}"
                )
            else:
                if mode == "drive":
                    action_name = apply_drive_action(gamepad, last_prediction)
                elif mode == "keyboard":
                    action_name = apply_keyboard_action(gamepad)

                print(f"[bridge] No prediction available yet mode={mode} action={action_name}", flush=True)
                gamepad.update()
                time.sleep(max(delay, 0.1))
                continue

            gamepad.update()
            time.sleep(delay)
    except KeyboardInterrupt:
        print("[bridge] Stopped EMG-to-game bridge.")
    finally:
        print("[bridge] Resetting gamepad and stopping classifier")
        gamepad.reset()
        gamepad.update()
        oclassi.stop_running()

def main():
    parser = argparse.ArgumentParser(description="Bridge real-time EMG predictions to a virtual gamepad.")
    parser.add_argument("--model_path", type=str, default=MODEL_PATH, help="Path to the saved .pkl model file")
    parser.add_argument("--majority-vote", type=int, default=config.MAJORITY_VOTE_WINDOW, help="Majority vote window size (1 disables it)")
    parser.add_argument("--delay", type=float, default=config.DELAY_BETWEEN_PREDICTIONS, help="Polling delay in seconds")
    parser.add_argument("--confidence-threshold", type=float, default=0.60, help="Minimum confidence required before applying a game action")
    parser.add_argument("--no-filter", action="store_true", help="Disable online notch/bandpass filters")
    args = parser.parse_args()

    run_realtime_game(
        args.model_path,
        majority_vote=args.majority_vote,
        delay=args.delay,
        use_filter=not args.no_filter,
        confidence_threshold=args.confidence_threshold,
    )


if __name__ == "__main__":
    main()
