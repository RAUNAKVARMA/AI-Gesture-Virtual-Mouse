import json
import logging
from pathlib import Path
from threading import Thread, Event
from typing import Optional

import cv2
import streamlit as st

import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from hand_tracking.hand_tracker import HandTracker  # type: ignore
from cursor_control.control_zone import ControlZone  # type: ignore
from cursor_control.mouse_controller import MouseController  # type: ignore
from gesture_recognition.gesture_classifier import GestureClassifier  # type: ignore
from gesture_recognition.gesture_logic import GestureEngine, GestureCommand  # type: ignore
from calibration.calibrator import Calibrator  # type: ignore


CONFIG_PATH = ROOT_DIR / "config" / "config.json"


def load_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_config(cfg: dict) -> None:
    with CONFIG_PATH.open("w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)


def create_engine_from_config(cfg: dict) -> tuple[HandTracker, MouseController, GestureEngine]:
    cam_cfg = cfg.get("camera", {})
    ht_cfg = cfg.get("hand_tracking", {})
    cz_cfg = cfg.get("control_zone", {})
    cursor_cfg = cfg.get("cursor", {})
    gest_cfg = cfg.get("gestures", {})

    control_zone = ControlZone(
        x_min=cz_cfg.get("x_min", 0.2),
        y_min=cz_cfg.get("y_min", 0.2),
        x_max=cz_cfg.get("x_max", 0.8),
        y_max=cz_cfg.get("y_max", 0.8),
    )

    tracker = HandTracker(
        camera_index=cam_cfg.get("index", -1),
        width=cam_cfg.get("width", 960),
        height=cam_cfg.get("height", 540),
        max_search_index=cam_cfg.get("max_search_index", 4),
        max_num_hands=ht_cfg.get("max_num_hands", 2),
        min_detection_confidence=ht_cfg.get("min_detection_confidence", 0.5),
        min_tracking_confidence=ht_cfg.get("min_tracking_confidence", 0.5),
        draw_landmarks=cfg.get("ui", {}).get("show_landmarks", True),
        show_control_zone=cfg.get("ui", {}).get("show_control_zone", True),
        control_zone=(
            control_zone.x_min,
            control_zone.y_min,
            control_zone.x_max,
            control_zone.y_max,
        ),
    )

    mouse = MouseController(
        control_zone=control_zone,
        sensitivity=cursor_cfg.get("sensitivity", 1.0),
        smoothing_factor=cursor_cfg.get("smoothing_factor", 0.5),
        enabled=cursor_cfg.get("enabled", True),
        demo_mode=gest_cfg.get("demo_mode", True),
    )

    classifier = GestureClassifier(
        model_path=gest_cfg.get("model_path", "ml/models/gesture_classifier.pkl"),
        use_model=gest_cfg.get("use_classifier", True),
    )
    calib_cfg = cfg.get("calibration", {}).get("thresholds", {})
    engine = GestureEngine(
        classifier=classifier,
        mouse=mouse,
        control_zone=control_zone,
        calibration_thresholds=calib_cfg,
    )
    return tracker, mouse, engine


def run_loop(stop_event: Event, status_placeholder, image_placeholder, gesture_placeholder) -> None:
    cfg = load_config()
    tracker, _, engine = create_engine_from_config(cfg)

    tracker.open()
    try:
        while not stop_event.is_set():
            ok, frame = tracker.read_frame()
            if not ok or frame is None:
                continue

            frame_out, hands = tracker.process(frame)
            command = engine.process(hands)

            if command:
                gesture_placeholder.markdown(f"**Gesture:** `{command.value}`")

            frame_rgb = cv2.cvtColor(frame_out, cv2.COLOR_BGR2RGB)
            image_placeholder.image(frame_rgb, channels="RGB")
            status_placeholder.markdown(
                f"**FPS:** `{tracker.fps:.1f}` &nbsp;&nbsp; **Hands:** `{len(hands)}`"
            )
    finally:
        tracker.close()


def run_calibration(status_placeholder) -> None:
    cfg = load_config()
    cam_cfg = cfg.get("camera", {})
    ht_cfg = cfg.get("hand_tracking", {})
    calib_cfg = cfg.get("calibration", {})

    tracker = HandTracker(
        camera_index=cam_cfg.get("index", -1),
        width=cam_cfg.get("width", 960),
        height=cam_cfg.get("height", 540),
        max_search_index=cam_cfg.get("max_search_index", 4),
        max_num_hands=ht_cfg.get("max_num_hands", 2),
        min_detection_confidence=ht_cfg.get("min_detection_confidence", 0.5),
        min_tracking_confidence=ht_cfg.get("min_tracking_confidence", 0.5),
    )
    calibrator = Calibrator(
        tracker=tracker,
        config_path=str(CONFIG_PATH),
        samples_per_gesture=calib_cfg.get("samples_per_gesture", 50),
    )
    status_placeholder.info("Running calibration in a separate OpenCV window. Press ESC to cancel.")
    calibrator.run()
    status_placeholder.success("Calibration finished. Reload dashboard to use new thresholds.")


def main() -> None:
    st.set_page_config(
        page_title="AI Gesture Virtual Mouse",
        layout="wide",
        page_icon="🖱️",
    )

    st.title("AI Gesture Virtual Mouse")
    st.caption("Production-style gesture-controlled virtual mouse with MediaPipe, OpenCV, and Streamlit.")

    cfg = load_config()

    if "controller_thread" not in st.session_state:
        st.session_state.controller_thread = None
        st.session_state.stop_event = Event()

    col_left, col_right = st.columns([2, 1])

    with col_left:
        image_placeholder = st.empty()
        status_placeholder = st.empty()
        gesture_placeholder = st.empty()

        col_buttons = st.columns(3)
        with col_buttons[0]:
            if st.button("▶ Start", type="primary"):
                if st.session_state.controller_thread is None or not st.session_state.controller_thread.is_alive():
                    st.session_state.stop_event.clear()
                    thread = Thread(
                        target=run_loop,
                        args=(st.session_state.stop_event, status_placeholder, image_placeholder, gesture_placeholder),
                        daemon=True,
                    )
                    st.session_state.controller_thread = thread
                    thread.start()
        with col_buttons[1]:
            if st.button("⏹ Stop"):
                if st.session_state.controller_thread is not None:
                    st.session_state.stop_event.set()
        with col_buttons[2]:
            if st.button("🎬 Demo mode toggle"):
                gest_cfg = cfg.get("gestures", {})
                gest_cfg["demo_mode"] = not gest_cfg.get("demo_mode", True)
                cfg["gestures"] = gest_cfg
                save_config(cfg)
                st.success(f"Demo mode set to {gest_cfg['demo_mode']}. Restart control loop to apply.")

    with col_right:
        st.subheader("Control Settings")
        cursor_cfg = cfg.get("cursor", {})
        sensitivity = st.slider(
            "Cursor sensitivity",
            min_value=0.2,
            max_value=3.0,
            value=float(cursor_cfg.get("sensitivity", 1.0)),
            step=0.1,
        )
        smoothing = st.slider(
            "Smoothing factor",
            min_value=0.0,
            max_value=1.0,
            value=float(cursor_cfg.get("smoothing_factor", 0.5)),
            step=0.05,
        )

        if st.button("Save settings"):
            cursor_cfg["sensitivity"] = sensitivity
            cursor_cfg["smoothing_factor"] = smoothing
            cfg["cursor"] = cursor_cfg
            save_config(cfg)
            st.success("Settings saved. Restart control loop to apply.")

        st.subheader("Calibration")
        calib_status = st.empty()
        if st.button("Run calibration"):
            if st.session_state.controller_thread is not None and st.session_state.controller_thread.is_alive():
                st.warning("Stop gesture control before running calibration.")
            else:
                Thread(target=run_calibration, args=(calib_status,), daemon=True).start()


if __name__ == "__main__":
    main()

