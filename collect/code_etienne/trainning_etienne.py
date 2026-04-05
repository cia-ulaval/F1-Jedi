import argparse
import libemg
from sklearn.metrics import accuracy_score
import datetime
from pathlib import Path


from config import Config
config = Config()

SUBJECT = config.SESSION_TRAIN

def filter_offline_data(odh:libemg.data_handler.OfflineDataHandler):
    fi = libemg.filtering.Filter(config.EMG_FS)
    fi.install_filters({"name": "notch", "cutoff": 60, "bandwidth": 3})
    fi.install_filters({"name": "bandpass", "cutoff": [20, 450], "order": 4})
    fi.filter(odh)

def load_data(subject, reps_train, reps_test, dataset_classes=None):
    session_values = [str(subject)]
    if dataset_classes is None:
        classes_values = [str(num) for num in range(config.NUM_CLASSES)]
    else:
        classes_values = [str(num) for num in dataset_classes]
    reps_values = [str(num) for num in range(config.NUM_REPS)]
    regex_filters = [
        libemg.data_handler.RegexFilter(left_bound="S", right_bound="/", values=session_values, description="session"),
        libemg.data_handler.RegexFilter(left_bound="C_", right_bound="_", values=classes_values, description="classes"),
        libemg.data_handler.RegexFilter(left_bound="R_", right_bound="_emg.csv", values=reps_values, description="reps")
    ]
    odh = libemg.data_handler.OfflineDataHandler()
    odh.get_data(config.DATA_PATH_TRAIN, regex_filters=regex_filters)
    
    filter_offline_data(odh)
    
    data_train = odh.isolate_data("reps", reps_train)
    data_test = odh.isolate_data("reps", reps_test)
    
    print(f"Data loaded. Training samples: {len(data_train.data)}, Testing samples: {len(data_test.data)}")
    
    return data_train, data_test

def extract_features(data, modality="emg"):
    window_size = int(config.WINDOW_SIZE_MS * config.EMG_FS / 1000)
    window_inc = int(config.WINDOW_INC_MS * config.EMG_FS / 1000)
    
    windows, metadata = data.parse_windows(window_size, window_inc)
    
    fe = libemg.feature_extractor.FeatureExtractor()
    # features = fe.extract_feature_group("LS4", windows)
    features_list = ["LS", "MFL", "MSR", "WAMP", "WENG"]
    features = fe.extract_features(feature_list=features_list, windows=windows)
    
    print(f"Extracted features from {len(windows)} windows.")
    
    return features, metadata["classes"]

def save_model(model:libemg.emg_predictor.EMGClassifier, subject):
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = Path(config.MODEL_PATH) / f"libemg_lda_S{subject}_{current_time}.pkl"
    model.save(model_path)
    print(f"Model saved at {model_path}")

def parse_reps_arg(raw_value):
    return [int(part.strip()) for part in raw_value.split(",") if part.strip()]

def train_and_test(session=None, train_reps=None, test_reps=None):
    subject = config.SESSION_COLLECT if session is None else session
    selected_train_reps = config.TRAIN_REPS if train_reps is None else train_reps
    selected_test_reps = config.TEST_REPS if test_reps is None else test_reps
    data_train, data_test = load_data(subject, selected_train_reps, selected_test_reps)
    
    features_train, labels_train = extract_features(data_train)
    features_test, labels_test = extract_features(data_test)
    
    model = libemg.emg_predictor.EMGClassifier('LDA')
    model.fit({"training_features": features_train, "training_labels": labels_train})
    
    predictions, _ = model.run(features_test)
    accuracy = accuracy_score(labels_test, predictions)
    
    print(f"Subject {subject} - Accuracy: {accuracy:.4f}")
    save_model(model, subject)
    print("Training and testing completed.")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate the Etienne LDA model.")
    parser.add_argument("--session", type=int, default=config.SESSION_COLLECT, help="Dataset session to load, e.g. 1 for dataset/S1")
    parser.add_argument("--train-reps", type=str, default=None, help="Comma-separated training repetitions, e.g. 2,0,4,1,5")
    parser.add_argument("--test-reps", type=str, default=None, help="Comma-separated testing repetitions, e.g. 3,6")
    args = parser.parse_args()
    train_and_test(
        session=args.session,
        train_reps=parse_reps_arg(args.train_reps) if args.train_reps else None,
        test_reps=parse_reps_arg(args.test_reps) if args.test_reps else None,
    )
