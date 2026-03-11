import argparse
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


def train(csv_path: str, output_model: str) -> None:
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path, header=None)

    # First column is label, rest are features
    y = df.iloc[:, 0].astype(str).values
    X = df.iloc[:, 1:].astype(float).values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    logger.info("Classification report:\n%s", classification_report(y_test, y_pred))

    output_model_path = Path(output_model)
    output_model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, output_model_path)
    logger.info("Saved trained model to %s", output_model_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train gesture classifier model.")
    parser.add_argument("--csv", required=True, help="Path to dataset CSV")
    parser.add_argument(
        "--output",
        default="ml/models/gesture_classifier.pkl",
        help="Output path for trained model",
    )
    args = parser.parse_args()
    train(args.csv, args.output)


if __name__ == "__main__":
    main()

