# F1-team-1
repo of the first team for the project F1

# 🚗 Gesture-Controlled Racing Car with EMG

## 📌 Project Overview
Take control of a miniature racing car using only hand gestures! This project utilizes an **electromyographic (EMG) bracelet** to capture muscle activity and convert it into driving commands. The system first operates in a simulation environment (such as **TrackMania + Pynput or vGamepad**), before transitioning to a real-world **F1 Tenth** car in collaboration with the **Véhicule Autonome de l’Université Laval (VAUL)**.

## 🎯 Key Features
- **EMG Signal Processing**: Converts muscle activity into driving inputs.
- **Gesture Recognition**: Maps hand movements to car controls.
- **Simulation Phase**: Testing in TrackMania + Pynput or vGamepad before real-world deployment.
- **Real-World Phase**: Controlling a physical **F1 Tenth** car.
- **Virtual Gamepad**: Sends EMG-based commands to the simulator through vGamepad.

## 🏗️ Project Pipeline
1. **Data Acquisition**: Collect EMG signals from the bracelet.
2. **Signal Processing**: Filter and analyze the signals to extract meaningful gestures.
3. **Command Mapping**: Translate gestures into acceleration, braking, and steering.
4. **Simulation Testing**: Validate the system in TrackMania + Pynput or vGamepad.
5. **Hardware Integration**: Deploy on the **F1 Tenth** car.
6. **Final Testing & Optimization**: Ensure real-time responsiveness and accuracy.

## 🛠️ Tech Stack
- **Python**: Core language for signal processing and interfacing.
- **Pynput**: Simulates keyboard inputs for the simulation.
- **vGamepad**: Simulates Joystick inputs for the simulation
- **TrackMania**: Virtual environment for initial testing.
- **F1 Tenth**: Real-world testing platform.
- **EMG Sensors**: Hardware for capturing muscle activity(Biopoint SifiLabs).

## 🚀 Getting Started
### Prerequisites
- Python 3.10+
- Windows for the game bridge (vgamepad needs the ViGEmBus driver)
- TrackMania
- SiFi Labs BioPoint (one EMG channel, 2000 Hz), read through libEMG

### Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/F1-team-1.git
cd F1-team-1

# Install dependencies
pip install -r requirements.txt
```

### Current EMG Workflow (3 gestures)
The BioPoint has a single EMG channel. With five gestures the decoder degraded within
minutes (skin temperature, sweat, muscle fatigue), so the pipeline now uses **three
gestures**: `No_Motion`, `Hand_Open`, `Wrist_Extension` (sensor on the dorsal forearm,
over the wrist extensors, 3-4 fingers below the elbow).

Which second gesture works depends on the person and on the sensor position. For the
first subject (sessions S1/S2) it was `Hand_Close`. For the current one (S3-S5), a normal
fist is barely above rest at this position (1.7x), while `Hand_Open` is ~10x and stays
stable when the sensor is removed and put back (S4 -> S5: 90.7 %, S5 -> S4: 97.4 %).
Measure it with `collect/code_etienne/live/live_gesture_survey.py` before choosing; the
trio is `GESTURES` in `config.py`.

```bash
# Activate the project virtual environment (Windows)
.\.venv\Scripts\Activate.ps1

# 1. Collect one session (3 gestures x 7 repetitions, about 3 minutes)
python .\collect\code_etienne\collect_etienne.py --session 6

# 2. Train and evaluate (one or several sessions)
python .\collect\code_etienne\trainning_etienne.py --sessions 4,5

# 3. Check the decoder in the console (latest model is loaded automatically)
python .\collect\code_etienne\realtime_test_etienne.py

# 4. Drive the game through the virtual gamepad
python .\collect\game.py
```

No sensor at hand? Replay a recorded session through the real-time loop, or run the tests:

```bash
python .\collect\code_etienne\realtime_test_etienne.py --replay 2
python -m pytest .\collect\code_etienne\tests -q
```

How the decoder works (`collect/code_etienne/emg_core.py`):
- **Stage 1, rest vs active**: window amplitude against a threshold placed between the rest
  level and the active level. The rest level is measured at start-up and tracked
  continuously, so the threshold follows the sensor.
- **Stage 2, which gesture**: LDA on normalized autocorrelation features. They do not depend
  on signal amplitude, so contraction strength and electrode gain do not change them.
- **Smoothing**: hysteresis plus a majority vote over 9 windows (about 180 ms).
- **Drift**: slow tracking of rest noise, amplitude and spectral shift, plus a 15-second
  recalibration (key `R` in the game bridge) when decoding degrades.

All settings live in `collect/code_etienne/config.py`. See `collect/COLLECT_PROTOCOL.md`
for the collection protocol.

## 🎮 Controls (drive mode, virtual Xbox 360 gamepad)
| Gesture | Action |
|---------|--------|
| `No_Motion` (relaxed hand) | Straight, accelerator held (button `A`) |
| `Hand_Open` (fingers extended and spread) | Steer left, accelerator held |
| `Wrist_Extension` (wrist lifted firmly, fingers relaxed) | Steer right, accelerator held |

The mapping is `GAME_ACTIONS` in `config.py` (swap left/right there, or use `brake`).
A weak wrist extension looks like `Hand_Open`: lift the wrist firmly. The decoder waits
300 ms at the start of each contraction before choosing a gesture (`ONSET_WINDOWS`),
which removed most wrong-way jerks in Mario Kart.

Live tests with the sensor (connection, signal quality, live accuracy, gesture survey,
offline replay of the recordings in `dataset/live/`) are in `collect/code_etienne/live/`.

### Mario Kart Wii on Dolphin
Map GameCube port 1 to `XInput/0/Gamepad` (A = `Button A`, B = `Button B`, main stick =
`Left X-/X+/Y-/Y+`), set Wii Remote 1 to None and enable Background Input. Start the bridge
first (it creates the virtual pad), then Dolphin; choose automatic drift in the game.

Keyboard keys while the bridge runs: `O` drive mode, `P` manual keyboard mode
(`V` = A, `B` = B, `G`/`J` = left/right, `Y`/`H` = up/down, `Space` = X), `R` quick
recalibration, `Q` quit. The gamepad is released if the sensor stops sending data.

## 📢 Future Enhancements
- Improve gesture detection using **machine learning**.
- Optimize signal filtering for real-time responsiveness.
- Enhance integration with **autonomous driving** features.

## 📜 License
This project is licensed under the MIT License.

## 🤝 Acknowledgments
- **Université Laval (VAUL)** for collaboration on F1 Tenth.
- Open-source tools like **TrackMania, Pynput**.

---
🚀 **Ready to race using just your hands? Let’s build the future of gesture-controlled vehicles!**

