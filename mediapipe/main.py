# lien de la documentation : https://developers.google.com/edge/mediapipe/solutions/vision/gesture_recognizer?

import mediapipe as mp

BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

def on_result(result, output_image: mp.Image, timestamp_ms: int):
    print(f"Gestures at {timestamp_ms} ms: {result.gestures}")

options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path="gesture_recognizer.task"),
    running_mode=VisionRunningMode.LIVE_STREAM,
    num_hands=2,
    result_callback=on_result,
)

with GestureRecognizer.create_from_options(options) as recognizer:
    # Async: automatically drops incoming camera frames when busy so results stay current
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    recognizer.recognize_async(mp_image, timestamp_ms)