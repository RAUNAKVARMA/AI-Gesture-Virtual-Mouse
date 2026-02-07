import json
import logging
from pathlib import Path

import cv2

from hand_tracking.hand_tracker import HandTracker
from cursor_control.control_zone import ControlZone
from cursor_control.mouse_controller import MouseController
from gesture_recognition.gesture_classifier import GestureClassifier
from gesture_recognition.gesture_logic import GestureEngine


def setup_logging(config: dict) -> None:
    log_cfg = config.get("logging", {})
    level_str = log_cfg.get("level", "INFO").upper()
    level = getattr(logging, level_str, logging.INFO)

    handlers = [logging.StreamHandler()]
    if log_cfg.get("log_to_file"):
        from logging.handlers import RotatingFileHandler

        handlers.append(
            RotatingFileHandler(
                log_cfg.get("filename", "gesture_virtual_mouse.log"),
                maxBytes=5 * 1024 * 1024,
                backupCount=2,
            )
        )

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        handlers=handlers,
    )


def load_config() -> dict:
    cfg_path = Path(__file__).resolve().parents[1] / "config" / "config.json"
    with cfg_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    config = load_config()
    setup_logging(config)
    logger = logging.getLogger("main")

    cam_cfg = config.get("camera", {})
    ht_cfg = config.get("hand_tracking", {})
    cz_cfg = config.get("control_zone", {})
    cursor_cfg = config.get("cursor", {})
    gest_cfg = config.get("gestures", {})
    calib_thresholds = config.get("calibration", {}).get("thresholds", {})

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
        draw_landmarks=config.get("ui", {}).get("show_landmarks", True),
        show_control_zone=config.get("ui", {}).get("show_control_zone", True),
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
    engine = GestureEngine(
        classifier=classifier,
        mouse=mouse,
        control_zone=control_zone,
        calibration_thresholds=calib_thresholds,
        cursor_hand_mode=gest_cfg.get("cursor_hand", "auto"),
    )

    logger.info("Starting AI Gesture Virtual Mouse")
    tracker.open()

    try:
        while True:
            ok, frame = tracker.read_frame()
            if not ok or frame is None:
                continue

            frame_out, hands = tracker.process(frame)
            command = engine.process(hands)

            y = 50
            ui_cfg = config.get("ui", {})

            # Overall state and gesture debug overlay
            if command is not None and ui_cfg.get("show_gesture_label", True):
                state_label = "ACTIVE" if engine.ctx.active else "PAUSED"
                cv2.putText(
                    frame_out,
                    f"State: {state_label}",
                    (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                )
                y += 25

                cv2.putText(
                    frame_out,
                    f"Gesture: {command.value} (conf={engine.ctx.gesture_confidence:.2f})",
                    (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 0),
                    2,
                )
                y += 25

                # Finger state array overlay
                if engine.ctx.last_finger_bits is not None:
                    t, i, m, r, p = engine.ctx.last_finger_bits
                    finger_str = f"Fingers [T,I,M,R,P]: [{t},{i},{m},{r},{p}]"
                    cv2.putText(
                        frame_out,
                        finger_str,
                        (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                    )
                    y += 25

            # Control zone / hand presence debug
            if not engine.ctx.hand_in_zone and engine.ctx.hand_present:
                cv2.putText(
                    frame_out,
                    "HAND OUTSIDE CONTROL ZONE",
                    (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )

            cv2.imshow("AI Gesture Virtual Mouse", frame_out)
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord("q"):
                break
    finally:
        tracker.close()
        cv2.destroyAllWindows()
        logger.info("Shut down cleanly")


if __name__ == "__main__":
    main()

