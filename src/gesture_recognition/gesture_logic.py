from collections import Counter, deque
from dataclasses import dataclass
from enum import Enum
from typing import Deque, Dict, List, Optional, Tuple

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
    last_raw_command: Optional[GestureCommand] = None
    gesture_confidence: float = 0.0
    active: bool = True
    hand_in_zone: bool = True
    hand_present: bool = False
    last_finger_state: Optional[Dict[str, bool]] = None
    last_finger_bits: Optional[Tuple[int, int, int, int, int]] = None
    debug_message: str = ""


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
        cursor_hand_mode: str = "auto",
    ) -> None:
        self.classifier = classifier
        self.mouse = mouse
        self.control_zone = control_zone
        self.ctx = GestureContext()
        self.calibration_thresholds = calibration_thresholds or {}
        # "auto" | "right" | "left" — which hand drives cursor when both visible
        self._cursor_hand_mode = (cursor_hand_mode or "auto").lower()

        # Temporal smoothing
        self._window: Deque[GestureCommand] = deque(maxlen=5)
        self._min_stable_frames = 3

    def process(self, hands: List[HandLandmarks]) -> Optional[GestureCommand]:
        """
        Process the current frame's hands and perform appropriate cursor actions.
        """
        if not hands:
            self.ctx.hand_present = False
            self.ctx.hand_in_zone = False
            self.ctx.debug_message = "NO HANDS DETECTED"
            self.ctx.last_raw_command = GestureCommand.PAUSE
            self._window.clear()
            self.ctx.last_command = GestureCommand.PAUSE
            return self.ctx.last_command

        right_hand = next((h for h in hands if h.hand_label.lower() == "right"), None)
        left_hand = next((h for h in hands if h.hand_label.lower() == "left"), None)

        self.ctx.hand_present = right_hand is not None or left_hand is not None

        cursor_hand, scroll_assist_hand = self._pick_hands(right_hand, left_hand)

        # Rule-based gesture decoding uses cursor hand (either hand when alone)
        raw_command = self._rule_based_command(cursor_hand, scroll_assist_hand)

        if raw_command is None:
            features = extract_features_multi_hand(hands)
            label_str = self.classifier.predict(features)
            try:
                raw_command = GestureCommand(label_str)
            except ValueError:
                raw_command = GestureCommand.PAUSE

        self.ctx.last_raw_command = raw_command

        # Temporal smoothing over the last N frames
        self._window.append(raw_command)
        counts = Counter(self._window)
        command, count = counts.most_common(1)[0]
        self.ctx.gesture_confidence = min(1.0, count / float(self._min_stable_frames))
        if count < self._min_stable_frames and self.ctx.last_command is not None:
            command = self.ctx.last_command

        # Control zone from the hand actually driving gestures
        if cursor_hand is not None:
            cx, cy = self.control_zone.center_of_hand(cursor_hand.landmarks)
            in_zone = self.control_zone.contains(cx, cy)
        else:
            cx, cy, in_zone = 0.5, 0.5, False

        self.ctx.hand_in_zone = in_zone
        if not in_zone:
            self.ctx.debug_message = "HAND OUTSIDE CONTROL ZONE"
        else:
            self.ctx.debug_message = ""

        # State machine: ACTIVE <-> PAUSED, but detection always runs
        if command == GestureCommand.PAUSE and self.ctx.active:
            self.ctx.active = True  # pause only affects actions, not detection
        if command == GestureCommand.RESUME:
            self.ctx.active = True

        # When paused (by closed fist), we still update context and smoothing
        # but we suppress cursor actions (movement / clicks / scroll).

        # Multi-hand semantics:
        # - Right hand: cursor movement + click/right-click
        # - Left hand: scroll and special gestures (via classifier outcome)
        actions_enabled = self.ctx.active and in_zone
        if command == GestureCommand.MOVE_CURSOR and cursor_hand is not None and actions_enabled:
            self._handle_move(cursor_hand)
        elif command == GestureCommand.LEFT_CLICK and actions_enabled:
            self.mouse.left_click()
        elif command == GestureCommand.RIGHT_CLICK and actions_enabled:
            self.mouse.right_click()
        elif command == GestureCommand.SCROLL and actions_enabled:
            # Two hands: optional finer scroll from non-cursor hand; else use cursor hand Y
            ref = scroll_assist_hand if scroll_assist_hand is not None else cursor_hand
            if ref is not None:
                dy = ref.landmarks[:, 1].mean() - 0.5
                self.mouse.scroll(-int(dy * 200))
            else:
                self.mouse.scroll(-50)

        self.ctx.last_command = command
        return command

    # ------------------------------------------------------------------
    # Rule-based gesture decoding utilities
    # ------------------------------------------------------------------

    def _pick_hands(
        self,
        right_hand: Optional[HandLandmarks],
        left_hand: Optional[HandLandmarks],
    ) -> Tuple[Optional[HandLandmarks], Optional[HandLandmarks]]:
        """
        Choose which hand drives cursor/gestures and which can assist scroll.

        - Single visible hand: always use it for cursor (fixes left-hand-only users).
        - Both visible: respect gestures.cursor_hand config (auto/right/left).
        Scroll assist = the other hand when two are present, else None.
        """
        if right_hand is None and left_hand is None:
            return None, None
        if right_hand is None:
            return left_hand, None
        if left_hand is None:
            return right_hand, None

        mode = self._cursor_hand_mode
        if mode == "left":
            cursor = left_hand
            assist = right_hand
        elif mode == "right":
            cursor = right_hand
            assist = left_hand
        else:
            # auto: default to right as primary when both (mirror-friendly)
            cursor = right_hand
            assist = left_hand

        return cursor, assist

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
            # Relaxed threshold so detection is robust across users / lighting
            state[name] = pts[tip, 1] < pts[pip, 1] - 0.01
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
        cursor_hand: Optional[HandLandmarks],
        _scroll_assist_hand: Optional[HandLandmarks],
    ) -> Optional[GestureCommand]:
        """
        Implement explicit gesture behaviors on the cursor hand (either hand when alone):
          - Index finger up → move cursor
          - Thumb + index pinch → left click
          - Two fingers → scroll
          - Three fingers → right click
          - Closed fist → pause tracking
          - Open palm → resume tracking
        """
        if cursor_hand is None:
            return GestureCommand.PAUSE

        right_state = self._finger_state(cursor_hand)
        pinch_dist = self._get_pinch_distance(cursor_hand)

        pinch_thresh = self._get_threshold("pinch", default=0.06)
        open_thresh = self._get_threshold("open_palm", default=0.12)
        fist_thresh = self._get_threshold("closed_fist", default=0.05)

        # Cache finger state for debug overlays
        bits = (
            int(right_state["thumb"]),
            int(right_state["index"]),
            int(right_state["middle"]),
            int(right_state["ring"]),
            int(right_state["pinky"]),
        )
        self.ctx.last_finger_state = right_state
        self.ctx.last_finger_bits = bits

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

    def _handle_move(self, hand: HandLandmarks) -> None:
        cx, cy = self.control_zone.center_of_hand(hand.landmarks)
        if not self.control_zone.contains(cx, cy):
            return
        self.mouse.move_from_normalized(cx, cy)


