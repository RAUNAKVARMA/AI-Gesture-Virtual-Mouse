from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class ControlZone:
    """
    Normalized control zone in camera space.
    Coordinates are in [0, 1] relative to frame size.
    """

    x_min: float = 0.2
    y_min: float = 0.2
    x_max: float = 0.8
    y_max: float = 0.8

    def contains(self, x: float, y: float) -> bool:
        return self.x_min <= x <= self.x_max and self.y_min <= y <= self.y_max

    def map_to_screen(
        self,
        x: float,
        y: float,
        screen_width: int,
        screen_height: int,
    ) -> Tuple[float, float]:
        """
        Map normalized camera coordinates within the control zone to screen pixels.
        """
        # Clamp to control zone
        x_clamped = min(max(x, self.x_min), self.x_max)
        y_clamped = min(max(y, self.y_min), self.y_max)

        # Re-normalize to [0,1] within zone
        x_norm = (x_clamped - self.x_min) / max(self.x_max - self.x_min, 1e-6)
        y_norm = (y_clamped - self.y_min) / max(self.y_max - self.y_min, 1e-6)

        return x_norm * screen_width, y_norm * screen_height

    def center_of_hand(self, landmarks: np.ndarray) -> Tuple[float, float]:
        """
        Compute the average x,y of given landmarks (assumed normalized).
        """
        if landmarks.size == 0:
            return 0.5, 0.5
        return float(landmarks[:, 0].mean()), float(landmarks[:, 1].mean())


