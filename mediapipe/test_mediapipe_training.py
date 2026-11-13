"""Entry point for custom gesture collection, training, and live recognition.

Examples:
    python test_mediapipe_training.py collect
    python test_mediapipe_training.py train
    python test_mediapipe_training.py live --width 1280 --height 720

Implementation lives in gesture_training/. This is not a pytest test.
"""

from gesture_training.cli import main


if __name__ == "__main__":
    main()
