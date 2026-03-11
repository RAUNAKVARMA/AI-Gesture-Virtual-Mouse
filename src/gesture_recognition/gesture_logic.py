from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

import numpy as np

from cursor_control.mouse_controller import MouseController
from cursor_control.control_zone import ControlZone
from hand_tracking.hand_tracker import HandLandmarks
from .feature_extractor import extract_features_multi_hand
from .gesture_classifier import GestureClassifier


class GestureCommand(str, Enum):
    MOVE_CURSOR = "MOVE_CURSOR"
    LEFT_CLICK = "LEFT_CLICK"
    RIGHT_CLICK = "RIGHT_CLICK"
    SCROLL = "SCROLL"
    PAUSE = "PAUSE"
    RESUME = "RESUME"


@dataclass
class GestureContext:
    last_command: Optional[GestureCommand] = None


class GestureEngine:
    """
    Orchestrates feature extraction, gesture classification and cursor actions.
    """

    def __init__(
        self,
        classifier: GestureClassifier,
        mouse: MouseController,
        control_zone: ControlZone,
        calibration_thresholds: Optional[Dict] = None,
    ) -> None:
        self.classifier = classifier
        self.mouse = mouse
        self.control_zone = control_zone
        self.ctx = GestureContext()
        self.calibration_thresholds = calibration_thresholds or {}

    def process(self, hands: List[HandLandmarks]) -> Optional[GestureCommand]:
        """
        Process the current frame's hands and perform appropriate cursor actions.
        """
        if not hands:
            self.ctx.last_command = GestureCommand.PAUSE
            return self.ctx.last_command

        # Separate major (right) and minor (left) hands if both are present
        right_hand = next((h for h in hands if h.hand_label.lower() == "right"), None)
        left_hand = next((h for h in hands if h.hand_label.lower() == "left"), None)

        # Rule-based gesture decoding from landmarks, optionally refined
        # by calibration thresholds; falls back to ML classifier if needed.
        command = self._rule_based_command(right_hand, left_hand)

        if command is None:
            features = extract_features_multi_hand(hands)
            label_str = self.classifier.predict(features)
            try:
                command = GestureCommand(label_str)
            except ValueError:
                command = GestureCommand.PAUSE

        # Multi-hand semantics:
        # - Right hand: cursor movement + click/right-click
        # - Left hand: scroll and special gestures (via classifier outcome)
        if command == GestureCommand.MOVE_CURSOR and right_hand is not None:
            self._handle_move(right_hand)
        elif command == GestureCommand.LEFT_CLICK:
            self.mouse.left_click()
        elif command == GestureCommand.RIGHT_CLICK:
            self.mouse.right_click()
        elif command == GestureCommand.SCROLL:
            # Scroll direction based on left-hand vertical movement if available
            if left_hand is not None:
                dy = left_hand.landmarks[:, 1].mean() - 0.5
                self.mouse.scroll(-int(dy * 200))
            else:
                self.mouse.scroll(-50)
        elif command == GestureCommand.PAUSE:
            # No movement or clicks
            pass
        elif command == GestureCommand.RESUME:
            # Just resume; next MOVE_CURSOR will begin motion again
            pass

        self.ctx.last_command = command
        return command

    # ------------------------------------------------------------------
    # Rule-based gesture decoding utilities
    # ------------------------------------------------------------------

    def _finger_state(self, hand: HandLandmarks) -> Dict[str, bool]:
        """
        Determine which fingers are up based on simple geometric rules.
        Uses y-coordinate ordering (in normalized coordinates, smaller y is higher).
        """
        pts = hand.landmarks
        tips = {  # tip, pip
            "thumb": (4, 3),
            "index": (8, 6),
            "middle": (12, 10),
            "ring": (16, 14),
            "pinky": (20, 18),
        }
        state: Dict[str, bool] = {}
        for name, (tip, pip) in tips.items():
            state[name] = pts[tip, 1] < pts[pip, 1] - 0.02
        return state

    def _get_pinch_distance(self, hand: HandLandmarks) -> float:
        pts = hand.landmarks
        return float(np.linalg.norm(pts[4, :2] - pts[8, :2]))

    def _get_threshold(self, gesture_name: str, default: float) -> float:
        """
        Read a scalar distance threshold from calibration, using first element
        of the stored mean vector as the relevant feature.
        """
        gesture_cfg = self.calibration_thresholds.get(gesture_name, {})
        mean_vec = gesture_cfg.get("mean")
        if isinstance(mean_vec, list) and len(mean_vec) > 0:
            try:
                return float(mean_vec[0])
            except (ValueError, TypeError):
                return default
        return default

    def _rule_based_command(
        self,
        right_hand: Optional[HandLandmarks],
        left_hand: Optional[HandLandmarks],
    ) -> Optional[GestureCommand]:
        """
        Implement explicit gesture behaviors:
          - Index finger up → move cursor
          - Thumb + index pinch → left click
          - Two fingers → scroll
          - Three fingers → right click
          - Closed fist → pause tracking
          - Open palm → resume tracking

        Right hand controls cursor; left hand provides scrolling / secondary gestures.
        """
        if right_hand is None:
            return GestureCommand.PAUSE

        right_state = self._finger_state(right_hand)
        pinch_dist = self._get_pinch_distance(right_hand)

        pinch_thresh = self._get_threshold("pinch", default=0.05)
        open_thresh = self._get_threshold("open_palm", default=0.15)
        fist_thresh = self._get_threshold("closed_fist", default=0.03)

        # Closed fist: all fingers down or very small spread
        if not any(right_state.values()) or pinch_dist < fist_thresh:
            return GestureCommand.PAUSE

        # Open palm: all fingers up and spread
        if all(right_state.values()) and pinch_dist > open_thresh:
            return GestureCommand.RESUME

        # Pinch (thumb + index together) → left click
        if pinch_dist < pinch_thresh:
            return GestureCommand.LEFT_CLICK

        # Two fingers (index + middle up) → scroll (handled via left hand if available)
        if (
            right_state["index"]
            and right_state["middle"]
            and not right_state["ring"]
            and not right_state["pinky"]
        ):
            return GestureCommand.SCROLL

        # Three fingers (index + middle + ring) → right click
        if (
            right_state["index"]
            and right_state["middle"]
            and right_state["ring"]
            and not right_state["pinky"]
        ):
            return GestureCommand.RIGHT_CLICK

        # Index finger up only → move cursor
        if (
            right_state["index"]
            and not right_state["middle"]
            and not right_state["ring"]
            and not right_state["pinky"]
        ):
            return GestureCommand.MOVE_CURSOR

        return None

    def _handle_move(self, right_hand: HandLandmarks) -> None:
        cx, cy = self.control_zone.center_of_hand(right_hand.landmarks)
        if not self.control_zone.contains(cx, cy):
            return
        self.mouse.move_from_normalized(cx, cy)


