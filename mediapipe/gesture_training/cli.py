"""Collect real landmarks, train five custom gestures, and test on a webcam.

Run with --help for commands. This is a manual experiment, not a pytest test.
Dependencies: mediapipe opencv-python numpy scikit-learn joblib
The trained artifact is a standalone classifier, NOT a MediaPipe .task bundle.
"""

import argparse
from pathlib import Path

from .config import ROOT
from .training import train
from .webcam import webcam


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("collect", "train", "live"))
    parser.add_argument("--camera", type=int, default=1)
    parser.add_argument("--width", type=int, default=1280, help="Preview/camera width")
    parser.add_argument("--height", type=int, default=720, help="Preview/camera height")
    parser.add_argument("--model", type=Path, default=ROOT / "gesture_recognizer.task")
    parser.add_argument("--data", type=Path, default=ROOT / "training_data" / "gestures.jsonl")
    parser.add_argument("--classifier", type=Path, default=ROOT / "models" / "custom_gestures.joblib")
    parser.add_argument("--interval", type=float, default=0.2, help="Seconds between saved samples")
    parser.add_argument("--threshold", type=float, default=0.75)
    parser.add_argument("--hold", type=float, default=0.2, help="Seconds of stable prediction")
    args = parser.parse_args()
    if args.interval <= 0 or not 0 <= args.threshold <= 1 or args.hold < 0:
        parser.error("Require interval > 0, threshold in [0,1], hold >= 0")
    if args.width <= 0 or args.height <= 0:
        parser.error("Require width > 0 and height > 0")
    try:
        train(args) if args.command == "train" else webcam(args)
    except (ValueError, OSError, ImportError) as error:
        parser.exit(1, f"Error: {error}\n")
