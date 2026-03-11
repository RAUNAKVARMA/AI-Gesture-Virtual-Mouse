from typing import List

import numpy as np

from .hand_tracker import HandLandmarks


def normalize_landmarks(
    hand: HandLandmarks,
    frame_width: int,
    frame_height: int,
) -> np.ndarray:
    """
    Normalize landmarks to pixel coordinates in [0, 1] relative to frame size.

    Returns:
        np.ndarray of shape (21, 3) with normalized (x, y, z).
    """
    coords = hand.landmarks.copy()
    # Convert normalized [0,1] to pixel space and re-normalize to [0,1] using
    # actual frame aspect ratio; this keeps coordinates resolution-independent.
    coords[:, 0] = (coords[:, 0] * frame_width) / frame_width
    coords[:, 1] = (coords[:, 1] * frame_height) / frame_height
    return coords


def stack_hands(hands: List[HandLandmarks], frame_width: int, frame_height: int) -> np.ndarray:
    """
    Flatten multiple hands into a single feature array.

    Layout:
        [right_hand(21*3), left_hand(21*3)]
    Missing hands are filled with zeros.
    """
    num_landmarks = 21 * 3
    right = np.zeros(num_landmarks, dtype=np.float32)
    left = np.zeros(num_landmarks, dtype=np.float32)

    for hand in hands:
        normalized = normalize_landmarks(hand, frame_width, frame_height).reshape(-1)
        if hand.hand_label.lower() == "right":
            right = normalized
        else:
            left = normalized

    return np.concatenate([right, left], axis=0)


