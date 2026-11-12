# Ajouts Ghita — Reconnaissance de gestes par vidéo (MediaPipe)

## Ce que j'ai ajouté
- `mediapipe/main.py` : lit la webcam (OpenCV), reconnaît les gestes de la main en temps réel avec **MediaPipe Gesture Recognizer**, et affiche les 21 points de la main, le geste reconnu, son score et les FPS.
- `requirements-vision.txt` : dépendances de la partie vidéo (séparées de la partie EMG).
- `.gitignore` : les modèles téléchargés (`*.task`, `*.pt`) ne sont pas versionnés, chacun les télécharge (voir plus bas).

## Comment ça marche (en bref)
1. OpenCV capture une image de la webcam (en miroir).
2. MediaPipe détecte jusqu'à 2 mains et donne 21 points 3D par main.
3. Le modèle classe le geste parmi : `Open_Palm`, `Closed_Fist`, `Thumb_Up`, `Thumb_Down`, `Victory`, `Pointing_Up`, `ILoveYou` (`None` sinon).
4. Le résultat est affiché à l'écran. Prochaine étape : associer chaque geste à une commande de jeu (pynput / vgamepad), comme dans `realtime_game_bridge.py`.

Prévu ensuite : MediaPipe **Hand Landmarker** (angle de la main pour la direction continue) + **YOLO** pour suivre la personne.

## Installation

Python **3.11 ou 3.12** requis (MediaPipe ne supporte pas toujours les versions plus récentes).

### Mac
```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements-vision.txt
```

### Windows (PowerShell)
```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-vision.txt
```

Si `requirements-vision.txt` pose problème (il a été généré sur Mac) :
```bash
pip install mediapipe opencv-python ultralytics numpy scikit-learn joblib matplotlib pynput
```

### Télécharger les modèles (Mac et Windows)
Sur Windows, remplacer `curl` par `curl.exe`.
```bash
curl -L -o mediapipe/gesture_recognizer.task https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/latest/gesture_recognizer.task
curl -L -o mediapipe/hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
python -c "from ultralytics import YOLO; YOLO('yolo11n.pt')"
```

## Lancer
```bash
python mediapipe/main.py
```
`q` ou `Échap` pour quitter.

## Problèmes fréquents
- **La webcam ne s'ouvre pas (Mac)** : Réglages Système > Confidentialité et sécurité > Caméra → autoriser Terminal / VS Code, puis relancer l'app.
- **Mauvaise caméra (ex. iPhone)** : mettre `CAMERA_INDEX = 1` en haut de `main.py`.
- **« Modèle introuvable »** : le fichier `.task` doit être dans le dossier `mediapipe/`.
- **vgamepad** ne fonctionne que sur Windows/Linux. Sur Mac, utiliser `pynput` (clavier).
