"""
Test MediaPipe Gesture Recognizer en temps réel avec la webcam.

Lancer depuis n'importe où (le chemin du modèle est résolu à partir de ce fichier) :
    python mediapipe/main.py

Touches :  q ou Échap = quitter

Doc : https://developers.google.com/edge/mediapipe/solutions/vision/gesture_recognizer/python
"""

import time
from pathlib import Path

import cv2
import mediapipe as mp

# --- Configuration -----------------------------------------------------------
HERE = Path(__file__).resolve().parent
MODEL_PATH = HERE / "gesture_recognizer.task"
CAMERA_INDEX = 0          # 0 = webcam du Mac ; essayer 1 si c'est un iPhone (Continuity Camera) qui s'ouvre
NUM_HANDS = 2
MIN_SCORE = 0.5           # en dessous, on affiche "?" au lieu du geste

# Les 21 points de la main sont reliés par ces segments (pour le dessin)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # pouce
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (5, 9), (9, 10), (10, 11), (11, 12),     # majeur
    (9, 13), (13, 14), (14, 15), (15, 16),   # annulaire
    (13, 17), (17, 18), (18, 19), (19, 20),  # auriculaire
    (0, 17),                                 # paume
]

BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


def draw_hand(frame, landmarks):
    """Dessine les 21 points et le squelette d'une main sur l'image BGR."""
    h, w = frame.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (255, 255, 255), 2)
    for x, y in pts:
        cv2.circle(frame, (x, y), 4, (0, 140, 255), -1)
    return pts


def main():
    if not MODEL_PATH.exists():
        raise SystemExit(
            f"Modèle introuvable : {MODEL_PATH}\n"
            "Télécharge-le avec :\n"
            "curl -L -o mediapipe/gesture_recognizer.task "
            "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/"
            "gesture_recognizer/float16/latest/gesture_recognizer.task"
        )

    options = GestureRecognizerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        # Mode VIDEO : appel synchrone, plus simple à déboguer que LIVE_STREAM + callback
        running_mode=VisionRunningMode.VIDEO,
        num_hands=NUM_HANDS,
    )

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise SystemExit(
            "Impossible d'ouvrir la webcam.\n"
            "Sur Mac : Réglages Système > Confidentialité et sécurité > Caméra, "
            "autorise Terminal / VS Code puis relance."
        )

    start = time.monotonic()
    last_ts = -1
    fps_t, fps = time.monotonic(), 0.0

    with GestureRecognizer.create_from_options(options) as recognizer:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Image webcam non reçue, arrêt.")
                break

            frame = cv2.flip(frame, 1)  # effet miroir, plus naturel
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            # Le timestamp doit être strictement croissant en mode VIDEO
            ts = int((time.monotonic() - start) * 1000)
            if ts <= last_ts:
                ts = last_ts + 1
            last_ts = ts

            result = recognizer.recognize_for_video(mp_image, ts)

            for i, landmarks in enumerate(result.hand_landmarks):
                pts = draw_hand(frame, landmarks)

                gesture, score = "?", 0.0
                if result.gestures and result.gestures[i]:
                    top = result.gestures[i][0]
                    gesture, score = top.category_name, top.score
                if score < MIN_SCORE:
                    gesture = "?"

                # Après le flip miroir, "Left"/"Right" de MediaPipe correspond à la main vue à l'écran
                side = result.handedness[i][0].category_name if result.handedness else ""

                label = f"{side} : {gesture} ({score:.2f})"
                x, y = pts[0]  # poignet
                cv2.putText(frame, label, (x - 60, y + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # FPS
            now = time.monotonic()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - fps_t, 1e-6))
            fps_t = now
            cv2.putText(frame, f"{fps:.0f} FPS", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

            cv2.imshow("F1-Jedi - MediaPipe Gesture Recognizer", frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
    