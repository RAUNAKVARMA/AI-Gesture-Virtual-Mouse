import csv
import logging
from pathlib import Path
from typing import List

import cv2

from gesture_recognition.feature_extractor import extract_features_multi_hand
from hand_tracking.hand_tracker import HandTracker


logger = logging.getLogger(__name__)


class DatasetBuilder:
    """
    Utility to record gesture samples for offline training.
    """

    def __init__(self, tracker: HandTracker, output_csv: str) -> None:
        self.tracker = tracker
        self.output_csv = Path(output_csv)

    def record(self, label: str, num_samples: int = 200) -> None:
        self.tracker.open()
        try:
            with self.output_csv.open("a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                collected = 0
                while collected < num_samples:
                    ok, frame = self.tracker.read_frame()
                    if not ok or frame is None:
                        continue
                    frame_out, hands = self.tracker.process(frame)
                    if not hands:
                        cv2.putText(
                            frame_out,
                            f"Show gesture '{label}' ({collected}/{num_samples})",
                            (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 0, 255),
                            2,
                        )
                        cv2.imshow("Dataset Builder", frame_out)
                        if cv2.waitKey(1) & 0xFF == 27:
                            break
                        continue

                    feats = extract_features_multi_hand(hands)
                    writer.writerow([label] + feats.astype(float).tolist())
                    collected += 1

                    cv2.putText(
                        frame_out,
                        f"Recording '{label}': {collected}/{num_samples}",
                        (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2,
                    )
                    cv2.imshow("Dataset Builder", frame_out)
                    if cv2.waitKey(1) & 0xFF == 27:
                        break
        finally:
            self.tracker.close()
            cv2.destroyWindow("Dataset Builder")


