from collections import deque
from typing import Deque, Optional, Tuple


class PositionSmoother:
    """
    Simple moving-average smoother for 2D cursor positions.
    """

    def __init__(self, window_size: int = 5, alpha: float = 0.5) -> None:
        self.window_size = max(1, window_size)
        self.alpha = min(max(alpha, 0.0), 1.0)
        self._buffer: Deque[Tuple[float, float]] = deque(maxlen=self.window_size)
        self._last: Optional[Tuple[float, float]] = None

    def reset(self) -> None:
        self._buffer.clear()
        self._last = None

    def update(self, x: float, y: float) -> Tuple[float, float]:
        """
        Add a new raw position and return the smoothed value.
        """
        if self._last is None:
            self._last = (x, y)
            self._buffer.append((x, y))
            return x, y

        # Exponential moving average on top of simple buffer to reduce jitter.
        ema_x = self.alpha * x + (1.0 - self.alpha) * self._last[0]
        ema_y = self.alpha * y + (1.0 - self.alpha) * self._last[1]
        self._last = (ema_x, ema_y)
        self._buffer.append(self._last)

        mean_x = sum(p[0] for p in self._buffer) / len(self._buffer)
        mean_y = sum(p[1] for p in self._buffer) / len(self._buffer)
        return mean_x, mean_y


