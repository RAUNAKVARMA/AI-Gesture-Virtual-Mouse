### AI Gesture Virtual Mouse

**AI Gesture Virtual Mouse** is a research‑grade computer vision system that turns your webcam into a virtual mouse.  
The mouse cursor is controlled using **hand gestures** detected by **MediaPipe Hands**, with stable landmark tracking, rule‑based logic, and an optional **ML classifier** for gesture recognition.  
The project is fully modular, production‑oriented, and ships with a **Streamlit dashboard** for interactive control and calibration.

---

### Features

- **Hand‑tracking with MediaPipe**
  - 21 landmarks per hand
  - Multi‑hand support (right + left)
  - Normalized coordinates and FPS overlay

- **Gesture‑driven mouse control**
  - **Index finger up** → move cursor
  - **Thumb + index pinch** → left click
  - **Two fingers up** → scroll
  - **Three fingers up** → right click
  - **Closed fist** → pause tracking
  - **Open palm** → resume tracking

- **Cursor control & smoothing**
  - PyAutoGUI‑based cursor, click, and scroll
  - Adjustable sensitivity and smoothing
  - Virtual **control zone** mapped from camera space to screen space

- **Calibration & configuration**
  - Interactive calibration for:
    - open palm
    - pinch
    - two fingers
    - closed fist
  - Thresholds stored in `config/config.json`
  - JSON‑driven configuration (camera, tracking, gestures, UI)

- **ML gesture classifier (Scikit‑learn)**
  - Dataset builder to record gesture samples
  - RandomForest‑based classifier (`ml/train_model.py`)
  - Loaded at runtime when available; falls back to robust rules

- **Streamlit dashboard**
  - Webcam preview
  - Start / Stop gesture control
  - Sensitivity & smoothing sliders
  - Calibration button
  - Gesture and FPS display
  - Demo mode (logs actions instead of moving the real cursor)

- **Production‑style engineering**
  - Modular architecture
  - Logging and error handling
  - Automatic camera detection
  - Clean separation between tracking, recognition, and control

---

### System Architecture

High‑level pipeline:

1. **Webcam (OpenCV)**
2. **Hand Tracking (MediaPipe Hands)**
3. **Landmark Feature Extraction**
4. **Gesture Recognition (rules + optional ML classifier)**
5. **Gesture Logic / State Machine**
6. **Cursor Controller (PyAutoGUI)**
7. **UI Layer (OpenCV overlay + Streamlit dashboard)**

---

### Project Structure

```text
AI-Gesture-Virtual-Mouse/
  src/
    hand_tracking/
      hand_tracker.py        # MediaPipe wrapper, FPS, overlays, multi-hand
      landmark_extractor.py  # Normalization & stacking utilities

    gesture_recognition/
      feature_extractor.py   # Landmark → feature vector
      gesture_classifier.py  # Scikit-learn wrapper + heuristics
      gesture_logic.py       # Rule-based gestures + multi-hand semantics

    cursor_control/
      mouse_controller.py    # PyAutoGUI integration, clicks, scroll
      smoothing.py           # Moving-average cursor smoothing
      control_zone.py        # Virtual control zone mapping

    calibration/
      calibrator.py          # Interactive calibration, writes config thresholds

    ml/
      dataset_builder.py     # Collect labeled gesture samples from webcam
      train_model.py         # Train RandomForest gesture classifier

    main.py                  # CLI entrypoint for the virtual mouse

  ui/
    dashboard.py             # Streamlit dashboard (preview, control, calibration)

  config/
    config.json              # Camera, tracking, gesture, UI, logging config

  assets/                    # (Optional) screenshots, diagrams, demo media

  requirements.txt
  README.md
  .gitignore
```

---

### Installation

#### 1. Clone the repository

```bash
git clone https://github.com/Viral-Doshi/Gesture-Controlled-Virtual-Mouse.git AI-Gesture-Virtual-Mouse
cd AI-Gesture-Virtual-Mouse
```

#### 2. Create and activate a virtual environment (recommended)

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux / macOS
```

#### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Required packages:

- `opencv-python`
- `mediapipe`
- `numpy`
- `pyautogui`
- `streamlit`
- `scikit-learn`

> **Note:** On some systems, `pyautogui` may require additional OS‑specific dependencies (e.g. accessibility permissions).

---

### Running the Core System (CLI)

From the project root:

```bash
python src/main.py
```

This will:

- auto‑detect a working camera if `camera.index` is set to `-1`
- open the OpenCV preview window
- show FPS, hand landmarks, and the control zone box
- respond to gestures:
  - **Index finger up** → move cursor within the green control zone
  - **Pinch (thumb + index)** → left click
  - **Two fingers** → scroll
  - **Three fingers** → right click
  - **Closed fist** → pause
  - **Open palm** → resume

Press `q` or `Esc` in the OpenCV window to exit.

---

### Running the Streamlit Dashboard

From the project root:

```bash
streamlit run ui/dashboard.py
```

Dashboard features:

- **Webcam preview**: live video with landmarks, control zone, and FPS.
- **Start / Stop gesture control**: run the recognition loop in a background thread.
- **Sensitivity slider**: adjust cursor movement sensitivity.
- **Smoothing slider**: control the EMA / moving-average smoothing factor.
- **Calibration button**: launches an OpenCV‑based calibration session.
- **Gesture display**: live label of the current interpreted gesture.
- **FPS counter**: real‑time performance indicator.

> Settings are persisted to `config/config.json`. After changing sliders or demo mode, stop and restart the loop (or reload the dashboard) to apply.

---

### Calibration Workflow

Calibration refines gesture thresholds using your own hand:

1. Ensure gesture control is **stopped** in the Streamlit dashboard.
2. Click **“Run calibration”**.
3. An OpenCV window will appear and step through:
   - `open_palm`
   - `pinch`
   - `two_fingers`
   - `closed_fist`
4. For each gesture:
   - Perform and hold the gesture in front of the camera.
   - Wait until the sample counter reaches the configured number (default 50).
   - Press `Esc` at any time to abort that gesture.
5. On completion:
   - Calibration statistics are written to `config/config.json` under `calibration.thresholds`.
   - The gesture engine will start using these thresholds (pinch distance, palm spread, fist tightness) on the next run.

You can also run calibration directly from the CLI, if desired, by creating a small script that instantiates `Calibrator` and calls `run()`.

---

### Gesture Classifier (ML)

The project includes a **Scikit‑learn** pipeline for data‑driven gesture classification:

- **Dataset collection**

  Use `DatasetBuilder` to record feature vectors for your custom gestures:

  ```bash
  # pseudocode example
  python -m src.ml.dataset_builder  # or write a small wrapper
  ```

  Each recorded row contains: `label, f1, f2, ..., fN`.

- **Model training**

  ```bash
  python src/ml/train_model.py --csv path/to/dataset.csv --output ml/models/gesture_classifier.pkl
  ```

  The trained RandomForest model is then loaded at runtime by `GestureClassifier`.

- **Runtime behavior**

  - If the model file exists and `gestures.use_classifier` is `true`, the classifier’s prediction is used.
  - If not, the system falls back to robust rule‑based logic and calibration thresholds.

---

### Configuration

All runtime configuration lives in `config/config.json`, for example:

- **camera**
  - `index`, `width`, `height`, `max_search_index`
- **hand_tracking**
  - MediaPipe parameters (max hands, detection/tracking confidence)
- **control_zone**
  - normalized coordinates of the active region in camera space
- **cursor**
  - `sensitivity`, `smoothing_factor`, `enabled`
- **gestures**
  - `model_path`, `demo_mode`, `use_classifier`
- **calibration**
  - `samples_per_gesture`, `thresholds` (populated by calibration)
- **logging**
  - log level, file logging options
- **ui**
  - toggles for FPS, landmarks, control zone, gesture label

Adjust these values to tune behavior without touching code.

---

### Future Improvements

- **Robust ML pipeline**
  - Better dataset tooling and augmentation
  - Support for per‑user profiles and online adaptation

- **Advanced gesture set**
  - Drag‑and‑drop, window management, multi‑finger shortcuts
  - Context‑aware gestures for different applications

- **Cross‑platform enhancements**
  - Improved macOS / Linux integration and permission helpers

- **Performance optimizations**
  - Async capture and inference
  - GPU‑accelerated pipelines and model serving

- **Testing & CI**
  - Unit tests for core modules
  - Continuous integration workflow for linting and checks

---

### Demo Section (Placeholder)

A curated **Demo** section can be added here with:

- GIFs or short videos of gesture control in action
- Screenshots of the Streamlit dashboard and calibration flow
- Links to blog posts or talks describing the system

Place media files under `assets/` and reference them from this section.

