"""Webcam preview, sample collection, and custom classifier recognition."""

from collections import Counter
import json
import time
import uuid

from .config import FEATURE_VERSION, gesture_names
from .features import features


def webcam(args):
    import cv2
    import mediapipe as mp
    import numpy as np

    from main import draw_hand_landmarks

    labels = gesture_names()
    artifact = None
    if args.command == "live":
        import joblib

        # joblib/pickle artifacts can execute code: load only files you trust.
        artifact = joblib.load(args.classifier)
        if (artifact.get("feature_version") != FEATURE_VERSION
                or artifact.get("labels") != labels):
            raise ValueError("Classifier feature format/labels differ; retrain it")
    if not args.model.is_file():
        raise ValueError(f"MediaPipe model missing: {args.model}")
    options = mp.tasks.vision.GestureRecognizerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(args.model)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=2,  # Reject ambiguous multi-hand collection/inference.
    )
    camera = cv2.VideoCapture(args.camera)
    session = uuid.uuid4().hex
    selected = None
    recording = False
    last_saved = 0.0
    last_timestamp = -1
    counts = Counter()
    candidate, candidate_since = "Unknown", 0.0
    source = None
    try:
        if not camera.isOpened():
            raise ValueError(f"Could not open camera {args.camera}")
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        window_name = "Custom gesture training experiment"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, args.width, args.height)
        if args.command == "collect":
            args.data.parent.mkdir(parents=True, exist_ok=True)
            source = args.data.open("a")
            print("Keys: 1-5 select class; R toggle recording; Q quit.")
            print("Session:", session)
            for index, label in enumerate(labels, 1):
                print(f"  {index}: {label}")
        with mp.tasks.vision.GestureRecognizer.create_from_options(options) as tracker:
            while True:
                success, frame = camera.read()
                if not success:
                    raise ValueError("Could not read webcam frame")
                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                timestamp = max(time.monotonic_ns() // 1_000_000, last_timestamp + 1)
                last_timestamp = timestamp
                result = tracker.recognize_for_video(
                    mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), timestamp
                )
                now = time.monotonic()
                hand = result.hand_landmarks[0] if len(result.hand_landmarks) == 1 else None
                vector = None
                coordinates = None
                if hand:
                    coordinates = [[lm.x, lm.y, lm.z] for lm in hand]
                    try:
                        vector = features(coordinates)
                    except ValueError:
                        pass
                if args.command == "collect":
                    if (recording and selected and vector is not None
                            and now - last_saved >= args.interval):
                        source.write(json.dumps({
                            "label": selected, "session": session,
                            "timestamp_ms": timestamp,
                            "handedness": result.handedness[0][0].category_name,
                            "mirrored": True, "landmarks": coordinates,
                        }) + "\n")
                        source.flush()
                        counts[selected] += 1
                        last_saved = now
                    hand_label = f"Training: {selected}" if selected else "Select 1-5"
                    status = f"{selected or 'Select 1-5'} | {'REC' if recording else 'PAUSED'}"
                    detail = f"Saved this run: {counts[selected]} | 1-5 select, R record, Q quit"
                else:
                    prediction, score = "Unknown", 0.0
                    if vector is not None:
                        probabilities = artifact["model"].predict_proba(vector.reshape(1, -1))[0]
                        best = int(np.argmax(probabilities))
                        score = float(probabilities[best])
                        if score >= args.threshold:
                            prediction = artifact["model"].classes_[best]
                    if prediction != candidate:
                        candidate, candidate_since = prediction, now
                    stable = (prediction if prediction != "Unknown"
                              and now - candidate_since >= args.hold else "Unknown")
                    hand_label = (f"{stable} ({score:.0%})"
                                  if stable != "Unknown" else "Unknown")
                    status = f"Gesture: {stable}"
                    detail = f"Candidate: {prediction} ({score:.0%}) | Q quit"
                if hand is None:
                    hand_label = "Unknown (show ONE hand)"
                    detail = "Show exactly ONE hand; missing/multiple hands = no action"
                # Use only custom classifier/collection labels, never built-in gestures.
                draw_hand_landmarks(
                    frame, result,
                    gesture_labels=[hand_label] * len(result.hand_landmarks),
                )
                cv2.rectangle(frame, (0, 0), (frame.shape[1], 85), (0, 0, 0), -1)
                for text, y in ((status, 30), (detail, 65)):
                    cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                                0.55, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.imshow(window_name, frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if args.command == "collect":
                    if ord("1") <= key <= ord("5"):
                        selected = labels[key - ord("1")]
                        recording = False  # Reposition before starting another class.
                    elif key == ord("r") and selected:
                        recording = not recording
    finally:
        if source is not None:
            source.close()
        camera.release()
        cv2.destroyAllWindows()
    if args.command == "collect":
        print("Saved this session:", dict(counts))
        print("Restart collection for a genuinely independent session before training.")
