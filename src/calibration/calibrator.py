import json
import logging
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

from gesture_recognition.feature_extractor import extract_features_multi_hand
from hand_tracking.hand_tracker import HandTracker, HandLandmarks


logger = logging.getLogger(__name__)


CALIBRATION_GESTURES = [
    "open_palm",
    "pinch",
    "two_fingers",
    "closed_fist",
]


class Calibrator:
    """
    Interactive calibration to compute gesture-specific thresholds.

    This is a lightweight helper to populate the `calibration.thresholds`
    section of `config.json`, which downstream components can optionally
    use for additional rule-based logic.
    """

    def __init__(
        self,
        tracker: HandTracker,
        config_path: str,
        samples_per_gesture: int = 50,
    ) -> None:
        self.tracker = tracker
        self.config_path = Path(config_path)
        self.samples_per_gesture = samples_per_gesture

    def _load_config(self) -> Dict:
        if not self.config_path.exists():
            return {}
        with self.config_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _save_config(self, cfg: Dict) -> None:
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with self.config_path.open("w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)

    def run(self) -> None:
        """
        Run calibration loop over predefined gestures.
        """
        cfg = self._load_config()
        calib_cfg = cfg.setdefault("calibration", {})
        thresholds_cfg = calib_cfg.setdefault("thresholds", {})

        self.tracker.open()
        try:
            for gesture_name in CALIBRATION_GESTURES:
                logger.info("Calibrating gesture: %s", gesture_name)
                feats = self._collect_samples(gesture_name)
                mean = feats.mean(axis=0)
                std = feats.std(axis=0) + 1e-6

                thresholds_cfg[gesture_name] = {
                    "mean": mean.tolist(),
                    "std": std.tolist(),
                }

            calib_cfg["samples_per_gesture"] = self.samples_per_gesture
            cfg["calibration"] = calib_cfg
            self._save_config(cfg)
            logger.info("Calibration complete and written to %s", self.config_path)
        finally:
            self.tracker.close()

    def _collect_samples(self, gesture_name: str) -> np.ndarray:
        collected: List[np.ndarray] = []
        while len(collected) < self.samples_per_gesture:
            ok, frame = self.tracker.read_frame()
            if not ok or frame is None:
                continue

            frame_out, hands = self.tracker.process(frame)
            if not hands:
                cv2.putText(
                    frame_out,
                    f"Show gesture: {gesture_name} ({len(collected)}/{self.samples_per_gesture})",
                    (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                )
                cv2.imshow("Calibration", frame_out)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
                continue

            feats = extract_features_multi_hand(hands)
            collected.append(feats)

            cv2.putText(
                frame_out,
                f"Capturing {gesture_name}: {len(collected)}/{self.samples_per_gesture}",
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )
            cv2.imshow("Calibration", frame_out)
            if cv2.waitKey(1) & 0xFF == 27:
                break

        cv2.destroyWindow("Calibration")
        return np.stack(collected, axis=0)


