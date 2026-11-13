# lien de la documentation : https://developers.google.com/edge/mediapipe/solutions/vision/gesture_recognizer?

# import mediapipe as mp
#
# BaseOptions = mp.tasks.BaseOptions
# GestureRecognizer = mp.tasks.vision.GestureRecognizer
# GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
# VisionRunningMode = mp.tasks.vision.RunningMode
#
# def on_result(result, output_image: mp.Image, timestamp_ms: int):
#     print(f"Gestures at {timestamp_ms} ms: {result.gestures}")
#
# options = GestureRecognizerOptions(
#     base_options=BaseOptions(model_asset_path="gesture_recognizer.task"),
#     running_mode=VisionRunningMode.LIVE_STREAM,
#     num_hands=2,
#     result_callback=on_result,
# )
#
# with GestureRecognizer.create_from_options(options) as recognizer:
#     # Async: automatically drops incoming camera frames when busy so results stay current
#     mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
#     recognizer.recognize_async(mp_image, timestamp_ms)


import time
from pathlib import Path
from threading import Lock

import cv2
import mediapipe as mp


MODEL_PATH = Path(__file__).resolve().parent / "gesture_recognizer.task"

# Wrist, thumb, fingers, and palm connections for the 21 hand landmarks.
HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
)
result_lock = Lock()
latest_hand_landmarks = ()
latest_gesture_labels = ()


def on_result(result, output_image: mp.Image, timestamp_ms: int):
    global latest_hand_landmarks, latest_gesture_labels

    # The callback runs asynchronously; share an immutable snapshot with the UI.
    with result_lock:
        latest_hand_landmarks = tuple(
            tuple((landmark.x, landmark.y) for landmark in hand)
            for hand in result.hand_landmarks
        )
        latest_gesture_labels = tuple(
            f"{gestures[0].category_name} ({gestures[0].score:.0%})"
            if gestures else "No gesture"
            for gestures in result.gestures
        )


def draw_hand_landmarks(frame, result=None, gesture_labels=None):
    """Draw landmarks, optionally using caller-provided gesture labels."""
    if result is None:
        with result_lock:
            hands = latest_hand_landmarks
            labels = latest_gesture_labels
    else:
        hands = tuple(
            tuple((landmark.x, landmark.y) for landmark in hand)
            for hand in result.hand_landmarks
        )
        labels = tuple(
            f"{gestures[0].category_name} ({gestures[0].score:.0%})"
            if gestures else "No gesture"
            for gestures in result.gestures
        )

    if gesture_labels is not None:
        labels = gesture_labels

    height, width = frame.shape[:2]
    for hand_index, hand in enumerate(hands):
        points = [ (int(x * (width - 1)), int(y * (height - 1))) for x, y in hand ]
        for start, end in HAND_CONNECTIONS:
            cv2.line(frame, points[start], points[end], (0, 255, 0), 2)
        for point in points:
            cv2.circle(frame, point, 4, (0, 0, 255), -1)

        if not points:
            continue
        label = labels[hand_index] if hand_index < len(labels) else "No gesture"
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.6
        thickness = 2
        (text_width, text_height), baseline = cv2.getTextSize(
            label, font, scale, thickness
        )
        x = max(0, min(min(p[0] for p in points), width - text_width - 8))
        y = max(text_height + 8, min(min(p[1] for p in points) - 12,
                                    height - baseline - 4))
        cv2.rectangle(
            frame, (x, y - text_height - 6),
            (x + text_width + 8, y + baseline + 4), (0, 0, 0), -1
        )
        cv2.putText(
            frame, label, (x + 4, y), font, scale,
            (255, 255, 255), thickness, cv2.LINE_AA
        )


def main():
    options = mp.tasks.vision.GestureRecognizerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=mp.tasks.vision.RunningMode.LIVE_STREAM,
        num_hands=2,
        result_callback=on_result,
    )

    camera = cv2.VideoCapture(1)

    try:
        if not camera.isOpened():
            raise RuntimeError("Could not open webcam.")

        with mp.tasks.vision.GestureRecognizer.create_from_options( options) as recognizer:
            last_timestamp_ms = -1

            while True:
                success, frame = camera.read()
                if not success:
                    print("Could not read a frame from the webcam.")
                    break

                # Mirror the image for a more natural webcam preview.
                frame = cv2.flip(frame, 1)

                # OpenCV uses BGR; MediaPipe expects RGB.
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image( image_format=mp.ImageFormat.SRGB, data=rgb_frame,)

                # MediaPipe requires strictly increasing timestamps.
                timestamp_ms = max( time.monotonic_ns() // 1_000_000, last_timestamp_ms + 1,)
                last_timestamp_ms = timestamp_ms

                recognizer.recognize_async(mp_image, timestamp_ms)

                # Draw only on the preview, not on the image sent to the model.
                # Live-stream results may trail the current frame slightly.
                draw_hand_landmarks(frame)
                cv2.imshow("Gestures + hand landmarks - press Q to quit", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

