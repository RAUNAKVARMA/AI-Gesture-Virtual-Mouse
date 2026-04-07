from __future__ import annotations

import copy
import json
import os
import sys
from io import BytesIO
from pathlib import Path
from threading import Thread
from typing import Any, Optional, Tuple

# Repo root on path so mediapipe_vision_stub can be imported when running `streamlit run ui/dashboard.py`
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
import mediapipe_vision_stub

mediapipe_vision_stub.apply()

import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image, ImageOps, ImageFile

# Partial JPEGs from some browsers / slow networks would otherwise decode as black/empty
ImageFile.LOAD_TRUNCATED_IMAGES = True

ROOT_DIR = _ROOT
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

_LAZY_DEPS: tuple[Any, ...] | None = None


def _lazy_deps() -> tuple[Any, ...]:
    """Import heavy gesture stack only when needed (faster Streamlit cold start)."""
    global _LAZY_DEPS
    if _LAZY_DEPS is None:
        from cursor_control.control_zone import ControlZone  # type: ignore
        from cursor_control.mouse_controller import MouseController  # type: ignore
        from gesture_recognition.gesture_classifier import GestureClassifier  # type: ignore
        from gesture_recognition.gesture_logic import GestureEngine  # type: ignore
        from hand_tracking.hand_tracker import HandTracker  # type: ignore

        _LAZY_DEPS = (HandTracker, ControlZone, MouseController, GestureClassifier, GestureEngine)
    return _LAZY_DEPS


CONFIG_PATH = ROOT_DIR / "config" / "config.json"

CAMERA_ERROR = "Camera not accessible"
CAMERA_FALLBACK_MESSAGE = (
    f"{CAMERA_ERROR}. Please allow permissions or run locally."
)


def open_local_webcam_capture(cfg: dict) -> Tuple[Optional[object], str]:
    """
    Open the first working webcam for server-side capture (Streamlit runs on your PC).
    Uses DirectShow on Windows for reliable access with OpenCV.
    Returns (cap_or_none, error_message).
    """
    import sys

    cv2 = __import__("cv2")
    cam_cfg = cfg.get("camera", {})
    configured = cam_cfg.get("index", -1)
    max_idx = max(cam_cfg.get("max_search_index", 4), 1)
    # Lower capture resolution speeds up driver init (processing size is separate).
    width = int(cam_cfg.get("capture_width", cam_cfg.get("width", 640)))
    height = int(cam_cfg.get("capture_height", cam_cfg.get("height", 360)))

    def try_index(idx: int):
        if sys.platform == "win32":
            cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
            if cap is not None and cap.isOpened():
                return cap
            if cap is not None:
                cap.release()
        cap = cv2.VideoCapture(idx)
        return cap if cap is not None and cap.isOpened() else None

    indices = [configured] if configured >= 0 else list(range(max_idx))
    last_err = ""
    for idx in indices:
        cap = try_index(idx)
        if cap is None or not cap.isOpened():
            last_err = f"Could not open camera index {idx}."
            continue
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        ok, frame = cap.read()
        if not ok or frame is None:
            cap.release()
            last_err = f"Camera index {idx} opened but returned no frames (in use or blocked?)."
            continue
        return cap, ""
    return None, last_err or CAMERA_FALLBACK_MESSAGE


def release_local_webcam_cap() -> None:
    cap = st.session_state.pop("gvm_local_cap", None)
    if cap is not None:
        try:
            cap.release()
        except Exception:
            pass


def ensure_local_opencv_engine(cfg: dict) -> tuple[Any, Any]:
    """
    MediaPipe + rule-based finger control (landmark 8 move, pinches for clicks).
    Classifier off so gestures are deterministic from MediaPipe Hands.
    """
    key = "gvm_local_opencv_engine"
    if key not in st.session_state:
        cfg_h = copy.deepcopy(cfg)
        cfg_h.setdefault("gestures", {})["use_classifier"] = False
        t, _m, engine = create_engine_from_config(cfg_h, use_opencv=True)
        t.open(use_camera=False)
        st.session_state[key] = (t, engine)
    return st.session_state[key]


def render_local_browser_camera_path(
    cfg: dict,
    status_placeholder,
    image_placeholder,
    gesture_placeholder,
) -> None:
    """Browser webcam via st.camera_input — triggers OS/browser permission prompts."""
    st.caption(
        "Click the camera area once so the browser can ask for **camera** access. "
        "Use **Chrome** or **Edge** for best results. Each capture refreshes detection."
    )
    tracker, engine = ensure_local_browser_tracker_engine(cfg)
    img_file = st.camera_input(
        "Webcam (browser)",
        key="gvm_local_browser_cam",
        help="Allow camera when prompted. If blocked, reset site permissions in the address bar.",
    )
    if img_file is None:
        status_placeholder.info("**Status:** waiting for a camera capture — click **Take photo** when ready.")
        return
    try:
        frame_rgb = _bytes_to_rgb_uint8(img_file.getvalue())
    except Exception:
        status_placeholder.error(CAMERA_FALLBACK_MESSAGE)
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


def _bytes_to_rgb_uint8(raw: bytes) -> np.ndarray:
    """
    Decode camera snapshot / upload bytes to HxWx3 uint8 RGB.
    Forces full decode and applies EXIF orientation (fixes sideways / blank-looking mobile shots).
    """
    img = Image.open(BytesIO(raw))
    img.load()
    img = ImageOps.exif_transpose(img)
    img = img.convert("RGB")
    arr = np.asarray(img, dtype=np.uint8)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    return np.ascontiguousarray(arr)


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


def ensure_hosted_tracker_engine(cfg: dict) -> tuple[Any, Any]:
    """One landmarker + engine per session (Streamlit Cloud / browser camera)."""
    key = "hosted_tracker_engine"
    if key not in st.session_state:
        cfg_h = copy.deepcopy(cfg)
        cfg_h.setdefault("gestures", {})["demo_mode"] = True
        tracker, _mouse, engine = create_engine_from_config(cfg_h, use_opencv=False)
        tracker.open(use_camera=False)
        st.session_state[key] = (tracker, engine)
    return st.session_state[key]


def ensure_local_browser_tracker_engine(cfg: dict) -> tuple[Any, Any]:
    """Browser camera on a local Streamlit run — respects config (e.g. demo_mode off for real mouse)."""
    key = "local_browser_tracker_engine"
    if key not in st.session_state:
        cfg_h = copy.deepcopy(cfg)
        tracker, _mouse, engine = create_engine_from_config(cfg_h, use_opencv=False)
        tracker.open(use_camera=False)
        st.session_state[key] = (tracker, engine)
    return st.session_state[key]


def _hosted_run_detection(
    frame_rgb: np.ndarray,
    mirror_horizontal: bool,
    tracker: Any,
    engine: Any,
    image_placeholder,
    gesture_placeholder,
    status_placeholder,
) -> None:
    if frame_rgb.size == 0 or frame_rgb.ndim != 3 or frame_rgb.shape[2] < 3:
        status_placeholder.error("Invalid image shape — try another photo.")
        return

    if float(frame_rgb.mean()) < 6.0:
        st.warning(
            "This frame looks **very dark** (almost black). Use brighter light, "
            "point the camera at yourself, or try **Upload image** with a clearer photo."
        )

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
        st.caption(
            "If the preview stays black: laptop **camera privacy shutter** off, no other app using the webcam, "
            "try **Edge** or **Firefox**, or use **Upload image** / run **`python src/main.py`** on your PC for full camera support."
        )

        rc1, rc2, rc3 = st.columns([1, 1, 1])
        with rc1:
            if st.button("Reset camera widget", key="gvm_reset_cam_btn", help="Remounts the camera if it got stuck"):
                st.session_state["gvm_cam_version"] = int(st.session_state.get("gvm_cam_version", 0)) + 1
                st.rerun()
        with rc2:
            st.link_button(
                "Test webcam elsewhere (new tab)",
                "https://webcamtests.com/check.html",
                use_container_width=True,
                help="If this page also shows no video, the issue is browser or Windows camera settings — not this app.",
            )
        with rc3:
            st.link_button(
                "Chrome camera settings",
                "https://support.google.com/chrome/answer/2693767",
                use_container_width=True,
            )

        with st.expander("Check browser camera permission (read-only)"):
            components.html(
                """
<div style="font-family:system-ui,sans-serif;font-size:14px;color:#333;">
  <button type="button" style="padding:8px 12px;cursor:pointer;border-radius:6px;"
    onclick="(async()=>{
      const o=document.getElementById('gvm-perm-out');
      o.textContent='Checking…';
      try{
        if(!navigator.permissions||!navigator.permissions.query){
          o.textContent='Permission API not available here (try the lock icon in the address bar → Site settings → Camera).';
          return;
        }
        const q=await navigator.permissions.query({name:'camera'});
        o.textContent='Camera permission state: '+q.state+
          '. If this is denied, click the lock icon next to the URL → Site settings → Camera → Allow, then reload.';
      }catch(e){
        o.textContent='Could not query: '+e.message+' — fix permissions via the lock icon in the address bar.';
      }
    })()">Show permission state</button>
  <pre id="gvm-perm-out" style="margin-top:10px;white-space:pre-wrap;font-size:13px;"></pre>
</div>
                """,
                height=200,
            )

        if "gvm_cam_version" not in st.session_state:
            st.session_state["gvm_cam_version"] = 0
        cam_widget_key = f"gvm_browser_camera_v{st.session_state['gvm_cam_version']}"

        img_file = st.camera_input(
            "Camera (click here to trigger the permission prompt if needed)",
            key=cam_widget_key,
            help="Browsers require a click on this widget before showing Allow/Block. If you blocked this site before, reset under the lock icon → Site settings → Camera.",
        )
        if img_file is None:
            status_placeholder.markdown(
                "**Status:** waiting for camera — or select **Upload image** above (always works)."
            )
            return

        tracker, engine = ensure_hosted_tracker_engine(cfg)
        try:
            frame_rgb = _bytes_to_rgb_uint8(img_file.getvalue())
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

    tracker, engine = ensure_hosted_tracker_engine(cfg)
    try:
        frame_rgb = _bytes_to_rgb_uint8(up.getvalue())
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
) -> tuple[Any, Any, Any]:
    HandTracker, ControlZone, MouseController, GestureClassifier, GestureEngine = _lazy_deps()
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
        show_fps=cfg.get("ui", {}).get("show_fps", True),
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


def render_local_opencv_live(
    cfg: dict,
    image_placeholder,
    status_placeholder,
    gesture_placeholder,
) -> None:
    """
    Auto-starts webcam, mirrors horizontally, downscales frames for speed, runs MediaPipe each tick.
    Camera + MediaPipe init run inside the fragment so the page shell renders immediately.
    """
    st.caption(
        "Webcam starts automatically. **Point** with your index finger to move the mouse; **pinch** thumb+index "
        "for left click, index+middle for right click. Run locally (not Streamlit Cloud)."
    )

    if not hasattr(st, "fragment"):
        status_placeholder.error(
            "Upgrade to Streamlit >= 1.33 for live video (`st.fragment`). " + CAMERA_FALLBACK_MESSAGE
        )
        return

    status_placeholder.info(
        "**Ready.** Loading camera and hand model in the background — video appears in a few seconds on first open."
    )

    cam_cfg = cfg.get("camera", {})
    proc_w = int(cam_cfg.get("process_width", 480))
    proc_h = int(cam_cfg.get("process_height", 270))

    @st.fragment(run_every=0.04)
    def _live_tick() -> None:
        import cv2

        if st.session_state.get("gvm_camera_failed") or st.session_state.get("gvm_hand_init_failed"):
            return

        if st.session_state.get("gvm_local_cap") is None:
            cap, _err = open_local_webcam_capture(cfg)
            if cap is None:
                st.session_state["gvm_camera_failed"] = True
                status_placeholder.error(CAMERA_ERROR)
                gesture_placeholder.empty()
                image_placeholder.empty()
                return
            st.session_state["gvm_local_cap"] = cap

        try:
            tracker, engine = ensure_local_opencv_engine(cfg)
        except Exception as exc:
            st.session_state["gvm_hand_init_failed"] = True
            status_placeholder.error(f"Hand tracking failed to start: `{exc}`")
            return

        cap = st.session_state.get("gvm_local_cap")
        if cap is None:
            return
        ok, frame = cap.read()
        if not ok or frame is None:
            status_placeholder.error(CAMERA_ERROR)
            return
        frame = cv2.flip(frame, 1)
        if proc_w > 0 and proc_h > 0:
            frame = cv2.resize(frame, (proc_w, proc_h), interpolation=cv2.INTER_AREA)
        try:
            frame_out, hands = tracker.process(frame)
        except Exception as exc:
            status_placeholder.error(f"Hand tracking failed: `{exc}`")
            return
        command = engine.process(hands)
        if command:
            gesture_placeholder.markdown(f"**Gesture:** `{command.value}`")
        frame_rgb = cv2.cvtColor(frame_out, cv2.COLOR_BGR2RGB)
        image_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
        status_placeholder.markdown(
            f"**FPS (processing):** `{tracker.fps:.1f}` · **Hands:** `{len(hands)}` · **Index tip → cursor**"
        )

    _live_tick()


def run_calibration(status_placeholder) -> None:
    from calibration.calibrator import Calibrator  # noqa: PLC0415 — local import avoids cv2 on Cloud

    HandTracker, *_rest = _lazy_deps()
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
            st.info(
                "Running on **your PC**: the camera opens automatically and **pyautogui** drives the system cursor."
            )
            if st.button("🎬 Toggle demo mode (no real mouse)", key="gvm_demo_toggle_local"):
                gest_cfg = cfg.get("gestures", {})
                gest_cfg["demo_mode"] = not gest_cfg.get("demo_mode", False)
                cfg["gestures"] = gest_cfg
                save_config(cfg)
                for k in ("gvm_local_opencv_engine", "local_browser_tracker_engine", "hosted_tracker_engine"):
                    st.session_state.pop(k, None)
                st.session_state.pop("gvm_hand_init_failed", None)
                st.session_state.pop("gvm_camera_failed", None)
                st.rerun()
            render_local_opencv_live(
                cfg, image_placeholder, status_placeholder, gesture_placeholder
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
            st.session_state.pop("gvm_local_opencv_engine", None)
            st.session_state.pop("gvm_hand_init_failed", None)
            st.success("Settings saved. Tracking engine will reload on the next video frame.")

        st.subheader("Calibration")
        if cloud_mode:
            st.caption(
                "Calibration opens a desktop OpenCV window — use a **local** install, not Streamlit Cloud."
            )
        calib_status = st.empty()
        if st.button("Run calibration"):
            if cloud_mode:
                st.error("Calibration is not available in Streamlit Cloud mode.")
            elif st.session_state.get("gvm_local_cap") is not None:
                st.warning(
                    "Close this Streamlit tab (or stop the server) so the webcam is free, then run calibration."
                )
            else:
                Thread(target=run_calibration, args=(calib_status,), daemon=True).start()


if __name__ == "__main__":
    main()

