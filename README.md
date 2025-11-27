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
- **Pynput Keyboard Simulation**: Sends EMG-based commands to the simulator.

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
- Python 3.8+
- TrackMania
- Pynput Library (`pip install pynput`)
- vGamepad Library(`pip install vgamepad`)
- EMG Sensor library (libEMG)

### Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/F1-team-1.git
cd F1-team-1

# Install dependencies
pip install -r requirements.txt
```

### Running the Simulation
```bash
python main.py
```

## 🎮 Controls for discrete actions (Mapped from EMG Signals) with pynput
| Gesture | Action |
|---------|--------|
| Thumb Flexion | Accelerate (Key: `W`) or (Key: `up` |
| Wrist Rotation Left | Turn Left (Key: `A` or (Key: `left`) |
| Wrist Rotation Right | Turn Right (Key: `D` or (Key: `right`) |
| Relaxed Hand | Brake (Key: `S`) or (Key: `down` |

## 🎮 Controls for continuous actions (Mapped from EMG Signals) with vgamepad
| Gesture | Action |
|---------|--------|
| Thumb Flexion | vgamepad.VX360Gamepad().right_trigger(255) |
| Wrist Rotation Left | vgamepad.VX360Gamepad().left_joystick(x_value=-32768, y_value=0) |
| Wrist Rotation Right | vgamepad.VX360Gamepad().left_joystick(x_value=+32768, y_value=0) |
| Relaxed Hand | vgamepad.VX360Gamepad().left_trigger(255)|



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

