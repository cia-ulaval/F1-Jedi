import argparse
import time
from pathlib import Path
import numpy as np
import libemg

from libemg.emg_predictor import OnlineEMGClassifier
from libemg.environments.controllers import ClassifierController

from config import Config
from collect_etienne import prepare_streamer


config = Config()

# MODEL_PATH = config.MODEL_PATH + "libemg_lda_S1_20241001_123456.pkl"  # Update with your actual model path
MODEL_PATH = None

def find_last_model(model_dir, subject):
    model_dir_path = Path(model_dir)
    if not model_dir_path.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir_path}")

    model_files = list(model_dir_path.glob(f"libemg_lda_S{subject}_*.pkl"))
    if not model_files:
        raise FileNotFoundError(f"No model files found for subject S{subject} in {model_dir_path}")

    latest_model = max(model_files, key=lambda f: f.stat().st_mtime)
    print(f"Latest model found: {latest_model}")
    return str(latest_model)

def load_model(model_path) -> libemg.emg_predictor.EMGClassifier:
    if model_path is None:
        model_path = find_last_model(config.MODEL_PATH, config.SESSION_TRAIN)
    model_path_obj = Path(model_path)
    if not model_path_obj.exists():
        raise FileNotFoundError(f"Model file not found: {model_path_obj}")

    classifier_cls = libemg.emg_predictor.EMGClassifier
    if hasattr(classifier_cls, "from_file"):
        model = classifier_cls.from_file(str(model_path_obj))
    else:
        raise AttributeError("libemg EMGClassifier has not from_file loader method.")

    print(f"Loaded model from: {model_path_obj}")
    return model


def wait_for_emg(odh, timeout_s=10.0):
    try:
        odh.reset()
    except Exception:
        pass

    t0 = time.perf_counter()
    while time.perf_counter() - t0 < timeout_s:
        vals, count = odh.get_data()
        try:
            total = int(np.asarray(count["emg"]).reshape(-1)[0])
        except Exception:
            total = 0
        if total > 0:
            return True
        time.sleep(0.05)
    return False


def map_class_label(pred_idx):
    if 0 <= pred_idx < len(config.CLASSES):
        return config.CLASSES[pred_idx]
    return pred_idx


def install_online_filters(odh:libemg.data_handler.OnlineDataHandler):
    fi = libemg.filtering.Filter(config.EMG_FS)
    fi.install_filters({"name": "notch", "cutoff": 60, "bandwidth": 3})
    fi.install_filters({"name": "bandpass", "cutoff": [20, 450], "order": 4})
    odh.install_filter(fi)


def run_realtime_test(model_path, majority_vote=config.MAJORITY_VOTE_WINDOW, delay=config.DELAY_BETWEEN_PREDICTIONS, use_filter=config.USE_FILTERS):
    window_size = int(config.WINDOW_SIZE_MS * config.EMG_FS / 1000)
    window_inc = int(config.WINDOW_INC_MS * config.EMG_FS / 1000)

    model = load_model(model_path)
    if majority_vote and majority_vote > 1:
        model.add_majority_vote(num_samples=majority_vote)

    odh = prepare_streamer(streaming=False)

    if use_filter:
        install_online_filters(odh)

    if not wait_for_emg(odh, timeout_s=config.TIMEOUT):
        raise RuntimeError(f"No EMG samples received from device after {config.TIMEOUT} seconds.")

    features_list = ["LS", "MFL", "MSR", "WAMP", "WENG"]

    num_classes = config.NUM_CLASSES
    try:
        num_classes = len(model.model.classes_)
    except Exception:
        pass

    oclassi = OnlineEMGClassifier(
        offline_classifier=model,
        window_size=window_size,
        window_inc=window_inc,
        online_data_handler=odh,
        features=features_list,
        std_out=False,
        output_format="probabilities",
    )
    controller = ClassifierController(
        output_format="probabilities",
        num_classes=num_classes,
    )

    try:
        odh.reset()
    except Exception:
        pass

    print("Starting real-time prediction. Press Ctrl+C to stop.")

    try:
        oclassi.run(block=False)
        while True:
            data = controller.get_data(["probabilities", "timestamp"])
            if data is not None:
                probabilities, timestamp = data
                probs = np.asarray(probabilities, dtype=float)
                pred_idx = int(np.argmax(probs))
                confidence = float(np.max(probs))
                print(
                    f"t={float(timestamp):.3f} pred_idx={pred_idx} mapped_label={map_class_label(pred_idx)} confidence={confidence:.3f}"
                )
            time.sleep(delay)
    except KeyboardInterrupt:
        print("Stopped real-time test.")
    finally:
        oclassi.stop_running()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Super-simple real-time EMG prediction test using a saved libemg model.")
    parser.add_argument("--model_path", type=str, default=MODEL_PATH, help="Path to the saved .pkl model file")
    parser.add_argument("--majority-vote", type=int, default=config.MAJORITY_VOTE_WINDOW, help="Majority vote window size (1 disables it)")
    parser.add_argument("--delay", type=float, default=config.DELAY_BETWEEN_PREDICTIONS, help="Polling delay in seconds")
    parser.add_argument("--no-filter", action="store_true", help="Disable online notch/bandpass filters")
    args = parser.parse_args()
    run_realtime_test(
        args.model_path,
        majority_vote=args.majority_vote,
        delay=args.delay,
        use_filter=not args.no_filter,
    )