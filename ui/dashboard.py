from __future__ import annotations

import copy
import json
import logging
import os
import queue
import sys
import time
from io import BytesIO
from pathlib import Path
from threading import Event, Thread
from typing import Any, Optional, Tuple

logger = logging.getLogger("gvm.dashboard")

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
    stop_gvm_capture_worker()
    cap = st.session_state.pop("gvm_local_cap", None)
    if cap is not None:
        try:
            cap.release()
        except Exception:
            pass


def _frame_queue_put(q: "queue.Queue", item: dict) -> None:
    """Keep only the newest frame(s): drop backlog if the UI is slow."""
    try:
        q.put_nowait(item)
    except queue.Full:
        try:
            while True:
                q.get_nowait()
        except queue.Empty:
            pass
        try:
            q.put_nowait(item)
        except queue.Full:
            pass


def stop_gvm_capture_worker() -> None:
    """Signal background capture thread to stop and release resources."""
    ev = st.session_state.get("gvm_worker_stop")
    if ev is not None:
        ev.set()
    th = st.session_state.get("gvm_worker_thread")
    if th is not None and th.is_alive():
        th.join(timeout=3.0)
    for key in ("gvm_worker_thread", "gvm_worker_stop", "gvm_frame_queue"):
        st.session_state.pop(key, None)


def _gesture_capture_worker(cfg: dict, stop_event: Event, frame_queue: "queue.Queue") -> None:
    """
    Background thread: OpenCV capture + MediaPipe + gesture/mouse.
    Never touches Streamlit — only pushes dicts to frame_queue.
    """
    import cv2

    cap = None
    tracker = None
    try:
        _frame_queue_put(frame_queue, {"kind": "status", "msg": "Worker: opening camera…"})
        logger.info("Worker: opening camera")
        cap, err_msg = open_local_webcam_capture(cfg)
        if cap is None or not cap.isOpened():
            logger.warning("Worker: camera failed: %s", err_msg)
            _frame_queue_put(frame_queue, {"kind": "error", "msg": CAMERA_ERROR})
            return

        _frame_queue_put(frame_queue, {"kind": "status", "msg": "Worker: loading hand model…"})
        logger.info("Worker: loading MediaPipe / hand model")
        cfg_h = copy.deepcopy(cfg)
        cfg_h.setdefault("gestures", {})["use_classifier"] = False
        tracker, _mouse, engine = create_engine_from_config(cfg_h, use_opencv=True)
        tracker.open(use_camera=False)

        cam_cfg = cfg.get("camera", {})
        proc_w = int(cam_cfg.get("process_width", 480))
        proc_h = int(cam_cfg.get("process_height", 270))

        logger.info("Worker: camera started, entering frame loop")
        _frame_queue_put(frame_queue, {"kind": "status", "msg": "Worker: streaming…"})

        while not stop_event.is_set():
            ok, frame = cap.read()
            if not ok or frame is None:
                logger.debug("Worker: skipped frame read")
                time.sleep(0.02)
                continue

            frame = cv2.flip(frame, 1)
            if proc_w > 0 and proc_h > 0:
                frame = cv2.resize(frame, (proc_w, proc_h), interpolation=cv2.INTER_AREA)

            try:
                frame_out, hands = tracker.process(frame)
            except Exception as exc:
                logger.exception("Worker: MediaPipe error (skipped frame): %s", exc)
                time.sleep(0.02)
                continue

            try:
                command = engine.process(hands)
            except Exception as exc:
                logger.exception("Worker: gesture engine error (skipped frame): %s", exc)
                time.sleep(0.02)
                continue

            frame_rgb = cv2.cvtColor(frame_out, cv2.COLOR_BGR2RGB)
            gtxt = str(command.value) if command else ""
            _frame_queue_put(
                frame_queue,
                {
                    "kind": "ok",
                    "rgb": frame_rgb,
                    "hands": len(hands),
                    "gesture": gtxt,
                    "fps": float(tracker.fps),
                },
            )
            time.sleep(0.001)

    except Exception as exc:
        logger.exception("Worker crashed: %s", exc)
        _frame_queue_put(frame_queue, {"kind": "error", "msg": f"Worker error: `{exc}`"})
    finally:
        try:
            if cap is not None:
                cap.release()
        except Exception:
            pass
        try:
            if tracker is not None:
                tracker.close()
        except Exception:
            pass
        logger.info("Worker: stopped")


def ensure_gvm_capture_worker(cfg: dict) -> None:
    """Start background worker once; survives Streamlit reruns while alive."""
    if st.session_state.get("gvm_disable_worker"):
        return
    th = st.session_state.get("gvm_worker_thread")
    if th is not None and th.is_alive():
        return
    stop_gvm_capture_worker()
    q: queue.Queue = queue.Queue(maxsize=2)
    ev = Event()
    st.session_state["gvm_frame_queue"] = q
    st.session_state["gvm_worker_stop"] = ev
    worker_cfg = copy.deepcopy(cfg)
    th = Thread(
        target=_gesture_capture_worker,
        args=(worker_cfg, ev, q),
        daemon=True,
        name="GVM-CaptureWorker",
    )
    st.session_state["gvm_worker_thread"] = th
    th.start()
    logger.info("Main: capture worker thread started")


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
    Main thread stays responsive: a background thread owns OpenCV + MediaPipe + gestures.
    The UI polls a queue inside st.fragment(run_every=...) and updates st.empty() placeholders.
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

    if st.session_state.get("gvm_disable_worker"):
        status_placeholder.warning(
            f"{CAMERA_ERROR} or worker stopped. Use **Toggle demo mode** (or **Save settings**) to retry."
        )
        return

    ensure_gvm_capture_worker(cfg)

    @st.fragment(run_every=0.05)
    def _poll_worker_queue() -> None:
        fq = st.session_state.get("gvm_frame_queue")
        if fq is None:
            return

        last_ok: dict | None = None
        last_status: str | None = None
        err_msg: str | None = None

        while True:
            try:
                item = fq.get_nowait()
            except queue.Empty:
                break
            kind = item.get("kind")
            if kind == "error":
                err_msg = item.get("msg", CAMERA_ERROR)
            elif kind == "status":
                last_status = item.get("msg", "")
            elif kind == "ok":
                last_ok = item
                err_msg = None

        if err_msg:
            st.session_state["gvm_disable_worker"] = True
            stop_gvm_capture_worker()
            status_placeholder.error(err_msg)
            logger.warning("UI: worker error, capture disabled until you retry: %s", err_msg)
            return

        if last_ok is not None:
            image_placeholder.image(
                last_ok["rgb"],
                channels="RGB",
                use_container_width=True,
            )
            if last_ok.get("gesture"):
                gesture_placeholder.markdown(f"**Gesture:** `{last_ok['gesture']}`")
            status_placeholder.markdown(
                f"**FPS:** `{last_ok['fps']:.1f}` · **Hands:** `{last_ok['hands']}` · **worker thread**"
            )
        elif last_status:
            status_placeholder.info(last_status)
        else:
            th = st.session_state.get("gvm_worker_thread")
            if th is not None and th.is_alive():
                status_placeholder.info("Waiting for camera / model (background thread)…")

    _poll_worker_queue()


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
    st.write("App started")

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
                stop_gvm_capture_worker()
                st.session_state.pop("gvm_disable_worker", None)
                for k in ("gvm_local_opencv_engine", "local_browser_tracker_engine", "hosted_tracker_engine"):
                    st.session_state.pop(k, None)
                st.session_state.pop("gvm_hand_init_failed", None)
                st.session_state.pop("gvm_camera_failed", None)
                st.session_state.pop("gvm_live_phase", None)
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
            stop_gvm_capture_worker()
            st.session_state.pop("gvm_disable_worker", None)
            st.session_state.pop("gvm_local_opencv_engine", None)
            st.session_state.pop("gvm_hand_init_failed", None)
            st.session_state.pop("gvm_live_phase", None)
            st.success("Settings saved. Restarting capture worker…")
            st.rerun()

        st.subheader("Calibration")
        if cloud_mode:
            st.caption(
                "Calibration opens a desktop OpenCV window — use a **local** install, not Streamlit Cloud."
            )
        calib_status = st.empty()
        if st.button("Run calibration"):
            if cloud_mode:
                st.error("Calibration is not available in Streamlit Cloud mode.")
            elif (
                (wt := st.session_state.get("gvm_worker_thread")) is not None
                and getattr(wt, "is_alive", lambda: False)()
            ):
                st.warning(
                    "Stop the live feed: toggle demo or save settings to restart the worker, then run calibration."
                )
            else:
                Thread(target=run_calibration, args=(calib_status,), daemon=True).start()


if __name__ == "__main__":
    main()

