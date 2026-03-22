import copy
import json
import logging
import os
import sys
from io import BytesIO
from pathlib import Path
from threading import Thread, Event
from typing import Optional

# Repo root on path so mediapipe_vision_stub can be imported when running `streamlit run ui/dashboard.py`
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
import mediapipe_vision_stub

mediapipe_vision_stub.apply()

import numpy as np
import streamlit as st
from PIL import Image

ROOT_DIR = _ROOT
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from hand_tracking.hand_tracker import HandTracker  # type: ignore
from cursor_control.control_zone import ControlZone  # type: ignore
from cursor_control.mouse_controller import MouseController  # type: ignore
from gesture_recognition.gesture_classifier import GestureClassifier  # type: ignore
from gesture_recognition.gesture_logic import GestureEngine, GestureCommand  # type: ignore


CONFIG_PATH = ROOT_DIR / "config" / "config.json"


def load_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_config(cfg: dict) -> None:
    with CONFIG_PATH.open("w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)


def use_streamlit_cloud_mode(cfg: dict) -> bool:
    """
    Browser-camera + landmarker-only path for Streamlit Community Cloud (no server webcam).

    Override with environment variable (recommended for Cloud secrets):
    - AI_GVM_STREAMLIT_CLOUD=1 / true / yes → cloud mode on
    - AI_GVM_STREAMLIT_CLOUD=0 / false / no → cloud mode off
    Otherwise uses config deployment.streamlit_cloud.
    """
    env = os.environ.get("AI_GVM_STREAMLIT_CLOUD", "").strip().lower()
    if env in ("1", "true", "yes", "on"):
        return True
    if env in ("0", "false", "no", "off"):
        return False
    return bool(cfg.get("deployment", {}).get("streamlit_cloud", False))


def ensure_hosted_tracker_engine(cfg: dict) -> tuple[HandTracker, GestureEngine]:
    """One landmarker + engine per session (Streamlit Cloud / browser camera)."""
    key = "hosted_tracker_engine"
    if key not in st.session_state:
        cfg_h = copy.deepcopy(cfg)
        cfg_h.setdefault("gestures", {})["demo_mode"] = True
        tracker, _mouse, engine = create_engine_from_config(cfg_h, use_opencv=False)
        tracker.open(use_camera=False)
        st.session_state[key] = (tracker, engine)
    return st.session_state[key]


def _hosted_run_detection(
    frame_rgb: np.ndarray,
    mirror_horizontal: bool,
    tracker: HandTracker,
    engine: GestureEngine,
    image_placeholder,
    gesture_placeholder,
    status_placeholder,
) -> None:
    if mirror_horizontal:
        frame_rgb = np.ascontiguousarray(frame_rgb[:, ::-1, :])
    else:
        frame_rgb = np.ascontiguousarray(frame_rgb)

    frame_out, hands = tracker.process_rgb(frame_rgb)
    command = engine.process(hands)

    if command:
        gesture_placeholder.markdown(f"**Gesture:** `{command.value}`")

    image_placeholder.image(frame_out, channels="RGB", use_container_width=True)
    status_placeholder.markdown(f"**Hands detected:** `{len(hands)}`")


def render_cloud_browser_camera_ui(
    cfg: dict,
    status_placeholder,
    image_placeholder,
    gesture_placeholder,
) -> None:
    st.info(
        "**Hosted mode:** Camera frames come from **your browser**. "
        "Hands are detected on the server. **PyAutoGUI is demo-only** here — "
        "clone the repo and run locally (`python src/main.py` or Streamlit without cloud mode) "
        "for real cursor control."
    )
    st.warning(
        "**Camera needs a real browser window** (not Cursor’s embedded preview). "
        "**Default below is “Upload image”** — no camera prompt required. "
        "For webcam, Chrome only shows **Allow / Block** after you **click** the camera area (or if the site wasn’t blocked before)."
    )
    with st.expander("Why don’t I see Chrome’s “Allow camera” (step 3)? Fix it"):
        st.markdown(
            """
**A. You must click the camera widget first**  
Chrome often shows the permission bar **only after** you click inside the gray camera box or **“Take photo”**. Nothing appears if you only scroll the page.

**B. You clicked “Block” earlier**  
Chrome will **not** ask again on that site. Reset it:
1. Click the **lock** or **tune** icon left of the address bar → **Site settings** → **Camera** → choose **Allow**.
2. Or open a new tab, paste: `chrome://settings/content/camera` → under **Blocked**, find `streamlit.app` (or your URL) → **delete** the block → reload this app.

**C. Wrong environment**  
Embedded IDE browsers block camera. **Copy this page’s URL** and open it in **Chrome** or **Edge** normally.

**Official help:** [Chrome camera & microphone](https://support.google.com/chrome/answer/2693767)
            """
        )

    tracker, engine = ensure_hosted_tracker_engine(cfg)

    input_mode = st.radio(
        "Choose input",
        [
            "Upload image (no camera — recommended)",
            "Webcam",
        ],
        horizontal=True,
        index=0,
        key="gvm_cloud_input_mode",
    )

    if input_mode == "Webcam":
        st.markdown(
            "**Next:** click once inside the camera area below — Chrome should then show **Allow while visiting the site** "
            "or **Block** at the top of the page (or in the address bar)."
        )
        img_file = st.camera_input(
            "Camera (click here to trigger the permission prompt if needed)",
            key="gvm_browser_camera",
            help="Browsers require a click on this widget before showing Allow/Block. If you blocked this site before, reset under the lock icon → Site settings → Camera.",
        )
        if img_file is None:
            status_placeholder.markdown(
                "**Status:** waiting for camera — or select **Upload image** above (always works)."
            )
            return

        try:
            img = Image.open(BytesIO(img_file.getvalue()))
            frame_rgb = np.array(img.convert("RGB"))
        except Exception:
            status_placeholder.error("Could not decode camera frame.")
            return

        _hosted_run_detection(
            frame_rgb,
            mirror_horizontal=True,
            tracker=tracker,
            engine=engine,
            image_placeholder=image_placeholder,
            gesture_placeholder=gesture_placeholder,
            status_placeholder=status_placeholder,
        )
        return

    st.caption("No camera permission required — works in Cursor’s preview and locked-down browsers.")
    up = st.file_uploader(
        "Choose a JPG / PNG / WebP with a hand in frame",
        type=["jpg", "jpeg", "png", "webp"],
        key="gvm_upload_frame",
    )
    if up is None:
        status_placeholder.markdown("**Status:** upload an image to run hand detection.")
        return

    try:
        img = Image.open(BytesIO(up.getvalue()))
        frame_rgb = np.array(img.convert("RGB"))
    except Exception:
        status_placeholder.error("Could not read that image.")
        return

    _hosted_run_detection(
        frame_rgb,
        mirror_horizontal=False,
        tracker=tracker,
        engine=engine,
        image_placeholder=image_placeholder,
        gesture_placeholder=gesture_placeholder,
        status_placeholder=status_placeholder,
    )


def create_engine_from_config(
    cfg: dict, *, use_opencv: bool = True
) -> tuple[HandTracker, MouseController, GestureEngine]:
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
        use_opencv=use_opencv,
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
        cursor_hand_mode=gest_cfg.get("cursor_hand", "auto"),
    )
    return tracker, mouse, engine


def run_loop(stop_event: Event, status_placeholder, image_placeholder, gesture_placeholder) -> None:
    import cv2

    cfg = load_config()
    tracker, _, engine = create_engine_from_config(cfg, use_opencv=True)

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
    from calibration.calibrator import Calibrator  # noqa: PLC0415 — local import avoids cv2 on Cloud

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

    cloud_mode = use_streamlit_cloud_mode(cfg)

    with col_left:
        image_placeholder = st.empty()
        status_placeholder = st.empty()
        gesture_placeholder = st.empty()

        if cloud_mode:
            render_cloud_browser_camera_ui(
                cfg, status_placeholder, image_placeholder, gesture_placeholder
            )
        else:
            col_buttons = st.columns(3)
            with col_buttons[0]:
                if st.button("▶ Start", type="primary"):
                    if st.session_state.controller_thread is None or not st.session_state.controller_thread.is_alive():
                        st.session_state.stop_event.clear()
                        thread = Thread(
                            target=run_loop,
                            args=(
                                st.session_state.stop_event,
                                status_placeholder,
                                image_placeholder,
                                gesture_placeholder,
                            ),
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
                    st.success(
                        f"Demo mode set to {gest_cfg['demo_mode']}. Restart control loop to apply."
                    )

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
        if cloud_mode:
            st.caption(
                "Calibration opens a desktop OpenCV window — use a **local** install, not Streamlit Cloud."
            )
        calib_status = st.empty()
        if st.button("Run calibration"):
            if cloud_mode:
                st.error("Calibration is not available in Streamlit Cloud mode.")
            elif st.session_state.controller_thread is not None and st.session_state.controller_thread.is_alive():
                st.warning("Stop gesture control before running calibration.")
            else:
                Thread(target=run_calibration, args=(calib_status,), daemon=True).start()


if __name__ == "__main__":
    main()

