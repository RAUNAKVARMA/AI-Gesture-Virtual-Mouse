import logging
from typing import Tuple

# Safe import for pyautogui (works locally, disabled on cloud)
try:
    import pyautogui
    MOUSE_AVAILABLE = True
except Exception:
    pyautogui = None
    MOUSE_AVAILABLE = False

from .smoothing import PositionSmoother
from .control_zone import ControlZone


logger = logging.getLogger(__name__)


class MouseController:
    """
    High-level cursor control using pyautogui.
    Automatically disables mouse control when running in environments
    without a GUI (e.g., Streamlit Cloud).
    """

    def __init__(
        self,
        control_zone: ControlZone,
        sensitivity: float = 1.0,
        smoothing_factor: float = 0.5,
        enabled: bool = True,
        demo_mode: bool = False,
    ) -> None:

        self.control_zone = control_zone
        self.sensitivity = sensitivity
        self.enabled = enabled and MOUSE_AVAILABLE
        self.demo_mode = demo_mode or not MOUSE_AVAILABLE

        if MOUSE_AVAILABLE:
            pyautogui.FAILSAFE = False
            screen_width, screen_height = pyautogui.size()
        else:
            # fallback values for cloud environments
            screen_width, screen_height = 1920, 1080

        self.screen_size: Tuple[int, int] = (screen_width, screen_height)
        self._refresh_screen_size_each_move = True
        self.smoother = PositionSmoother(window_size=5, alpha=smoothing_factor)

        logger.info(
            "MouseController initialized (screen=%sx%s, sensitivity=%.2f, smoothing=%.2f, enabled=%s, demo_mode=%s)",
            screen_width,
            screen_height,
            self.sensitivity,
            smoothing_factor,
            self.enabled,
            self.demo_mode,
        )

        if not MOUSE_AVAILABLE:
            logger.warning("Mouse control disabled (no display environment detected).")


    def _apply_sensitivity(self, x: float, y: float) -> Tuple[float, float]:

        if not MOUSE_AVAILABLE:
            return x, y

        cx, cy = pyautogui.position()

        dx = (x - cx) * self.sensitivity
        dy = (y - cy) * self.sensitivity

        return cx + dx, cy + dy


    def move_from_normalized(self, x_norm: float, y_norm: float) -> None:
        """
        Move cursor from a normalized camera coordinate in [0,1]x[0,1].
        Maps using the current screen size from pyautogui.size() when available.
        """

        if not self.enabled:
            return

        if MOUSE_AVAILABLE and self._refresh_screen_size_each_move:
            self.screen_size = pyautogui.size()

        x, y = self.control_zone.map_to_screen(
            x_norm,
            y_norm,
            int(self.screen_size[0]),
            int(self.screen_size[1]),
        )

        x, y = self._apply_sensitivity(x, y)

        x, y = self.smoother.update(x, y)

        if self.demo_mode:
            logger.debug("Demo move cursor to (%.1f, %.1f)", x, y)
            return

        pyautogui.moveTo(int(round(x)), int(round(y)), duration=0.0)


    def left_click(self) -> None:

        if not self.enabled:
            return

        if self.demo_mode:
            logger.info("Demo left click")
            return

        pyautogui.click(button="left")


    def right_click(self) -> None:

        if not self.enabled:
            return

        if self.demo_mode:
            logger.info("Demo right click")
            return

        pyautogui.click(button="right")


    def scroll(self, dy: int) -> None:
        """
        Scroll vertically by dy units (positive=up, negative=down).
        """

        if not self.enabled:
            return

        if self.demo_mode:
            logger.info("Demo scroll dy=%d", dy)
            return

        pyautogui.scroll(dy)

