# Collect, train, and recognize custom gestures

Use this tool to record hand gestures with your webcam, train a classifier,
and test its predictions live. It currently supports the first five gestures
in `gesture_list.json`.

Run all commands below from the `mediapipe/` directory.

## 1. Prepare your environment

Install the required packages in your Python environment:

```bash
python -m pip install mediapipe opencv-python numpy scikit-learn joblib
```

Make sure `gesture_recognizer.task` is present in the `mediapipe/` directory.
This MediaPipe model detects hand landmarks; the classifier you train below
provides the custom gesture names.

The default camera index is **1**. If the wrong camera opens, or no camera opens,
try `--camera 0`. Allow camera access in your operating system if prompted.

## 2. Collect training samples

```bash
python test_mediapipe_training.py collect --camera 1
```

Show **exactly one hand** to the camera, then use these controls:

| Key | Action |
| --- | --- |
| `1` | Select `No_Motion` |
| `2` | Select `Hand_Close` |
| `3` | Select `Hand_Open` |
| `4` | Select `Wrist_Flexion` |
| `5` | Select `Wrist_Extension` |
| `R` | Start or pause recording for the selected gesture |
| `Q` | Quit and close the camera |

For each gesture:

1. Press its number key. Recording starts paused whenever you select a class.
2. Position your hand in the correct pose.
3. Press `R` to record. The preview shows `REC` and the saved sample count.
4. Vary your hand position, distance, and lighting slightly while keeping the
   gesture correct.
5. Press `R` again to pause before changing poses.

Samples are appended to `training_data/gestures.jsonl`; existing samples are
not overwritten. By default, one sample is saved every **0.2 seconds** while
recording and valid landmarks are available.

### Record multiple independent sessions

Record every gesture, quit with `Q`, and restart collection to create another
session. Change your position or lighting between runs so evaluation measures
more than nearly identical frames.

Training requires at least **20 samples per gesture across at least two
separate runs**. Aim for more samples and several runs covering all five
gestures: the training/validation split must also have sufficient examples of
every class on both sides. Adjacent frames from one run are never split between
training and evaluation.

To change the sampling interval:

```bash
python test_mediapipe_training.py collect --interval 0.3
```

## 3. Train the classifier

Close the collection window, then run:

```bash
python test_mediapipe_training.py train
```

The command prints a classification report and confusion matrix for held-out
sessions, then saves:

- `models/custom_gestures.joblib` — the trained custom classifier.
- `models/custom_gestures.metrics.json` — evaluation results.

Training again replaces these output files. The classifier is not a MediaPipe
`.task` bundle; live recognition uses it alongside `gesture_recognizer.task`.

If training reports insufficient samples or cannot find a valid
session-separated split, collect all five gestures in more independent runs
and try again.

## 4. Test recognition live

```bash
python test_mediapipe_training.py live --camera 1
```

Show one hand and hold a gesture steady. The hand label displays your **trained
classifier's gesture name and confidence**, not MediaPipe's built-in gesture
names. Press `Q` to quit.

By default, a prediction must reach **75% confidence** and remain the same for
**0.2 seconds** before its gesture label appears. Otherwise, the label is
`Unknown`. Missing or multiple hands also produce no accepted gesture.

Adjust these settings if needed:

```bash
python test_mediapipe_training.py live --threshold 0.85 --hold 0.4
```

A higher threshold or longer hold time makes recognition more conservative.
Only load classifier files you trust: joblib files can execute Python code.

## Preview size

The default preview is a resizable **1280 × 720** window. Set another size for
collection or live recognition with:

```bash
python test_mediapipe_training.py live --width 1600 --height 900
```

The tool also requests this capture resolution from the camera, but the actual
capture resolution depends on what your camera supports.

## Use different data or model files

```bash
python test_mediapipe_training.py collect --data training_data/session_set.jsonl
python test_mediapipe_training.py train --data training_data/session_set.jsonl --classifier models/my_gestures.joblib
python test_mediapipe_training.py live --classifier models/my_gestures.joblib
```

Use `--model path/to/gesture_recognizer.task` to select a different MediaPipe
landmark model. Default resource paths are resolved relative to `mediapipe/`;
relative paths you supply are resolved from your current working directory.

For all available options:

```bash
python test_mediapipe_training.py --help
```
