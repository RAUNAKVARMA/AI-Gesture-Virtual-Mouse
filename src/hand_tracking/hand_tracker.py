import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class HandLandmarks:
    hand_label: str  # "Left" or "Right"
    landmarks: np.ndarray  # shape (21, 3) in normalized coordinates (x, y, z)


class HandTracker:
    """
    MediaPipe Tasks-based hand tracker with camera management, FPS tracking,
    visualization, and multi-hand support.

    Set ``use_opencv=False`` for environments where OpenCV cannot load (e.g. some
    Streamlit Cloud images missing GLib). Use :meth:`process_rgb` with RGB frames
    from Pillow / browser camera in that case. Local desktop use keeps
    ``use_opencv=True`` (default).
    """

    MODEL_URL = (
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
        "hand_landmarker/float16/1/hand_landmarker.task"
    )

    def __init__(
        self,
        camera_index: int = -1,
        width: int = 960,
        height: int = 540,
        max_search_index: int = 4,
        max_num_hands: int = 2,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        draw_landmarks: bool = True,
        show_control_zone: bool = True,
        control_zone: Tuple[float, float, float, float] = (0.2, 0.2, 0.8, 0.8),
        model_path: Optional[str] = None,
        use_opencv: bool = True,
    ) -> None:
        # Direct imports avoid loading vision/__init__.py (drawing_utils -> cv2).
        # Streamlit Cloud: ui/dashboard applies mediapipe_vision_stub first.
        from mediapipe.tasks.python.core.base_options import BaseOptions
        from mediapipe.tasks.python.vision.core.vision_task_running_mode import (
            VisionTaskRunningMode,
        )
        from mediapipe.tasks.python.vision.hand_landmarker import (
            HandLandmarker,
            HandLandmarkerOptions,
        )

        self._BaseOptions = BaseOptions
        self._HandLandmarker = HandLandmarker
        self._HandLandmarkerOptions = HandLandmarkerOptions
        self._VisionTaskRunningMode = VisionTaskRunningMode

        self.use_opencv = use_opencv
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.max_search_index = max_search_index
        self.max_num_hands = max_num_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self.cap: Optional[Any] = None
        self._landmarker: Optional[object] = None

        self.draw_landmarks = draw_landmarks
        self.show_control_zone = show_control_zone
        self.control_zone = control_zone  # (x_min, y_min, x_max, y_max) in [0,1]

        self._prev_time = time.time()
        self.fps: float = 0.0

        root = Path(__file__).resolve().parents[2]
        default_model = root / "assets" / "hand_landmarker.task"
        self.model_path = Path(model_path) if model_path is not None else default_model

    def _cv2(self):
        import cv2

        return cv2

    def _ensure_model(self) -> None:
        if self.model_path.exists():
            return
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Downloading MediaPipe hand model to %s", self.model_path)
        try:
            import urllib.request

            urllib.request.urlretrieve(self.MODEL_URL, self.model_path.as_posix())
        except Exception as exc:
            logger.exception("Failed to download hand_landmarker model: %s", exc)
            raise RuntimeError(
                "Unable to download MediaPipe hand_landmarker model. "
                f"Please download it manually from {self.MODEL_URL} "
                f"and place it at {self.model_path}"
            ) from exc

    def _create_landmarker(self) -> None:
        self._ensure_model()
        base_options = self._BaseOptions(model_asset_path=str(self.model_path))
        options = self._HandLandmarkerOptions(
            base_options=base_options,
            running_mode=self._VisionTaskRunningMode.IMAGE,
            num_hands=self.max_num_hands,
            min_hand_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
        )
        self._landmarker = self._HandLandmarker.create_from_options(options)

    def _detect_camera_index(self) -> int:
        """Automatically detect a working camera index."""
        cv2 = self._cv2()
        if self.camera_index >= 0:
            logger.info("Using configured camera index %d", self.camera_index)
            return self.camera_index

        logger.info("Auto-detecting camera index up to %d", self.max_search_index)
        for idx in range(self.max_search_index):
            cap = cv2.VideoCapture(idx)
            if cap is not None and cap.isOpened():
                ret, _ = cap.read()
                cap.release()
                if ret:
                    logger.info("Detected working camera at index %d", idx)
                    return idx
        logger.error("No working camera found in indices [0, %d)", self.max_search_index)
        return 0

    def open(self, use_camera: bool = True) -> None:
        """
        Open the camera stream (optional) and initialize the hand landmarker.

        Set ``use_camera=False`` for Streamlit Cloud / browser-camera mode: only
        the landmarker is created; feed frames via ``process_rgb()`` or ``process()``.
        """
        if use_camera:
            if not self.use_opencv:
                raise RuntimeError("Local camera requires use_opencv=True (install opencv-python).")
            cv2 = self._cv2()
            index = self._detect_camera_index()
            self.cap = cv2.VideoCapture(index)
            if not self.cap or not self.cap.isOpened():
                raise RuntimeError(f"Failed to open camera at index {index}")

            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            logger.info("Camera opened at index %d (%dx%d)", index, self.width, self.height)
        else:
            self.cap = None
            logger.info("HandTracker opened without local camera (browser/external frames only)")

        self._create_landmarker()

    def close(self) -> None:
        """Release resources."""
        if self.cap:
            self.cap.release()
            self.cap = None
        if self._landmarker:
            self._landmarker.close()
            self._landmarker = None
        logger.info("Camera and MediaPipe resources released")

    def _update_fps(self) -> None:
        current = time.time()
        dt = max(current - self._prev_time, 1e-6)
        self._prev_time = current
        self.fps = 1.0 / dt

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read a frame from the camera."""
        if not self.cap:
            raise RuntimeError(
                "No local camera: open with use_camera=True, or pass frames to process() "
                "(e.g. from st.camera_input)."
            )

        cv2 = self._cv2()
        success, frame = self.cap.read()
        if not success or frame is None:
            logger.warning("Failed to read frame from camera")
            return False, None

        frame = cv2.flip(frame, 1)
        return True, frame

    def process(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, List[HandLandmarks]]:
        """
        Run MediaPipe on a BGR frame (OpenCV path). Requires ``use_opencv=True``.
        """
        if not self.use_opencv:
            raise RuntimeError("Use process_rgb() when use_opencv=False")

        from mediapipe.tasks.python.vision.core.image import Image, ImageFormat

        if self._landmarker is None:
            raise RuntimeError("HandTracker.open() must be called before process()")

        self._update_fps()

        cv2 = self._cv2()
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = Image(image_format=ImageFormat.SRGB, data=frame_rgb)

        result = self._landmarker.detect(mp_image)

        hands_out: List[HandLandmarks] = []
        if result is not None and result.hand_landmarks:
            for landmarks_list, handedness_list in zip(
                result.hand_landmarks, result.handedness
            ):
                label = handedness_list[0].category_name
                coords = np.array(
                    [[lm.x, lm.y, lm.z] for lm in landmarks_list],
                    dtype=np.float32,
                )
                hands_out.append(HandLandmarks(hand_label=label, landmarks=coords))

                if self.draw_landmarks:
                    self._draw_points(frame_bgr, coords)

        if self.show_control_zone:
            self._draw_control_zone(frame_bgr)

        self._draw_fps(frame_bgr)
        return frame_bgr, hands_out

    def process_rgb(self, frame_rgb: np.ndarray) -> Tuple[np.ndarray, List[HandLandmarks]]:
        """
        Run MediaPipe on an RGB uint8 frame without OpenCV (Streamlit Cloud).

        Draws overlays with Pillow. Returns RGB image with same shape.
        """
        from mediapipe.tasks.python.vision.core.image import Image, ImageFormat
        from PIL import Image as PILImage
        from PIL import ImageDraw

        if self.use_opencv:
            raise RuntimeError("process_rgb is only for use_opencv=False")

        if self._landmarker is None:
            raise RuntimeError("HandTracker.open() must be called before process_rgb()")

        self._update_fps()

        if frame_rgb.dtype != np.uint8:
            frame_rgb = frame_rgb.astype(np.uint8)
        if not frame_rgb.flags["C_CONTIGUOUS"]:
            frame_rgb = np.ascontiguousarray(frame_rgb)

        h, w = frame_rgb.shape[:2]
        mp_image = Image(image_format=ImageFormat.SRGB, data=frame_rgb)
        result = self._landmarker.detect(mp_image)

        pil_img = PILImage.fromarray(frame_rgb, mode="RGB")
        draw = ImageDraw.Draw(pil_img)

        hands_out: List[HandLandmarks] = []
        if result is not None and result.hand_landmarks:
            for landmarks_list, handedness_list in zip(
                result.hand_landmarks, result.handedness
            ):
                label = handedness_list[0].category_name
                coords = np.array(
                    [[lm.x, lm.y, lm.z] for lm in landmarks_list],
                    dtype=np.float32,
                )
                hands_out.append(HandLandmarks(hand_label=label, landmarks=coords))

                if self.draw_landmarks:
                    for x, y, _ in coords:
                        cx = int(x * w)
                        cy = int(y * h)
                        r = 3
                        draw.ellipse(
                            (cx - r, cy - r, cx + r, cy + r),
                            fill=(255, 255, 0),
                            outline=(255, 255, 0),
                        )

        if self.show_control_zone:
            x_min, y_min, x_max, y_max = self.control_zone
            x0, y0 = int(x_min * w), int(y_min * h)
            x1, y1 = int(x_max * w), int(y_max * h)
            draw.rectangle([x0, y0, x1, y1], outline=(0, 255, 0), width=2)

        draw.text((10, 10), f"FPS: {self.fps:.1f}", fill=(255, 255, 255))

        out = np.array(pil_img)
        return out, hands_out

    def _draw_points(self, frame_bgr: np.ndarray, coords: np.ndarray) -> None:
        cv2 = self._cv2()
        h, w, _ = frame_bgr.shape
        for x, y, _ in coords:
            cx = int(x * w)
            cy = int(y * h)
            cv2.circle(frame_bgr, (cx, cy), 3, (0, 255, 255), -1)

    def _draw_control_zone(self, frame_bgr: np.ndarray) -> None:
        cv2 = self._cv2()
        h, w, _ = frame_bgr.shape
        x_min, y_min, x_max, y_max = self.control_zone
        p1 = (int(x_min * w), int(y_min * h))
        p2 = (int(x_max * w), int(y_max * h))
        cv2.rectangle(frame_bgr, p1, p2, (0, 255, 0), 2)

    def _draw_fps(self, frame_bgr: np.ndarray) -> None:
        cv2 = self._cv2()
        cv2.putText(
            frame_bgr,
            f"FPS: {self.fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
