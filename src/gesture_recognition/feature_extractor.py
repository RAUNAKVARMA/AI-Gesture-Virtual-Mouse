from typing import List

import numpy as np

from hand_tracking.hand_tracker import HandLandmarks


FINGER_TIPS = [4, 8, 12, 16, 20]


def pairwise_distances(points: np.ndarray, indices: List[int]) -> List[float]:
    dists: List[float] = []
    for i in range(len(indices)):
        for j in range(i + 1, len(indices)):
            p1 = points[indices[i]]
            p2 = points[indices[j]]
            dists.append(float(np.linalg.norm(p1[:2] - p2[:2])))
    return dists


def finger_angles(points: np.ndarray) -> List[float]:
    """
    Rough angles for index and middle fingers using three joints each.
    """
    angles: List[float] = []
    triplets = [
        (5, 6, 8),   # index
        (9, 10, 12), # middle
    ]
    for a, b, c in triplets:
        v1 = points[a, :2] - points[b, :2]
        v2 = points[c, :2] - points[b, :2]
        denom = (np.linalg.norm(v1) * np.linalg.norm(v2)) + 1e-6
        cos_angle = float(np.clip(np.dot(v1, v2) / denom, -1.0, 1.0))
        angles.append(cos_angle)
    return angles


def extract_features_for_hand(hand: HandLandmarks) -> np.ndarray:
    """
    Convert a single hand's landmarks into a feature vector suitable for ML.
    """
    pts = hand.landmarks
    feats: List[float] = []

    # Distances between fingertips (gesture shape)
    feats.extend(pairwise_distances(pts, FINGER_TIPS))

    # Basic finger joint angles
    feats.extend(finger_angles(pts))

    # Relative fingertip positions to wrist (landmark 0)
    wrist = pts[0, :2]
    for tip_idx in FINGER_TIPS:
        delta = pts[tip_idx, :2] - wrist
        feats.extend(delta.tolist())

    return np.asarray(feats, dtype=np.float32)


def extract_features_multi_hand(hands: List[HandLandmarks]) -> np.ndarray:
    """
    Concatenate features for right and left hands into a single vector.
    Missing hands are filled with zeros so the feature size is fixed.
    """
    right_feat: np.ndarray = np.zeros(0, dtype=np.float32)
    left_feat: np.ndarray = np.zeros(0, dtype=np.float32)

    for hand in hands:
        feats = extract_features_for_hand(hand)
        if hand.hand_label.lower() == "right":
            right_feat = feats
        else:
            left_feat = feats

    if right_feat.size == 0:
        right_feat = np.zeros_like(left_feat)
    if left_feat.size == 0:
        left_feat = np.zeros_like(right_feat)

    return np.concatenate([right_feat, left_feat], axis=0)


