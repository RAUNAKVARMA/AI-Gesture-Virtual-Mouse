import logging
from pathlib import Path
from typing import List, Optional

import joblib
import numpy as np


logger = logging.getLogger(__name__)


GESTURE_LABELS = [
    "MOVE_CURSOR",
    "LEFT_CLICK",
    "RIGHT_CLICK",
    "SCROLL",
    "PAUSE",
    "RESUME",
]


class GestureClassifier:
    """
    Thin wrapper around a scikit-learn classifier.

    If no model file is available, it falls back to a very simple
    rule-based heuristic suitable for demo mode.
    """

    def __init__(self, model_path: str, use_model: bool = True) -> None:
        self.model_path = Path(model_path)
        self.use_model = use_model
        self._model: Optional[object] = None

        if self.use_model and self.model_path.exists():
            try:
                self._model = joblib.load(self.model_path)
                logger.info("Loaded gesture classifier from %s", self.model_path)
            except Exception as exc:
                logger.exception("Failed to load gesture classifier: %s", exc)
                self._model = None
        else:
            if self.use_model:
                logger.warning(
                    "Gesture classifier model not found at %s. Falling back to heuristics.",
                    self.model_path,
                )

    def predict(self, features: np.ndarray) -> str:
        """
        Predict the gesture label given a single feature vector.
        """
        if self._model is not None:
            try:
                pred = self._model.predict(features.reshape(1, -1))[0]
                return str(pred)
            except Exception as exc:
                logger.exception("Model prediction failed: %s", exc)

        # Heuristic / demo fallback
        return self._heuristic_predict(features)

    def _heuristic_predict(self, features: np.ndarray) -> str:
        """
        Very simple rule-based logic that makes the system usable even
        without a trained ML model.
        """
        if features.size < 10:
            return "PAUSE"

        # Fingertip distance heuristics on right hand (first few distances)
        d1 = float(features[0])
        d2 = float(features[1]) if features.size > 1 else d1

        # Closed pinch (small distance) -> click
        if d1 < 0.05:
            return "LEFT_CLICK"

        # Larger spread -> move
        if d1 > 0.15:
            return "MOVE_CURSOR"

        # Intermediate -> scroll
        if 0.05 <= d1 <= 0.15:
            return "SCROLL"

        return "PAUSE"


