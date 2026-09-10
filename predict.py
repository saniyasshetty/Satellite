"""
predict.py
----------
Command-line prediction on a single image.

Run:
    python predict.py path/to/image.png
"""

import sys
import joblib
from features import extract_features
MODEL_PATH = "poverty_model.joblib"


def predict(image_path):
    bundle = joblib.load(MODEL_PATH)
    clf = bundle["model"]

    feats = extract_features(image_path).reshape(1, -1)
    pred = clf.predict(feats)[0]
    proba = clf.predict_proba(feats)[0]
    # IMPORTANT: proba columns follow clf.classes_ order (alphabetical),
    # NOT the order the classes were listed in during training.
    classes = clf.classes_

    print(f"\nImage: {image_path}")
    print(f"Predicted Poverty Level: {pred.upper()}\n")
    print("Confidence breakdown:")
    for cls, p in sorted(zip(classes, proba), key=lambda x: -x[1]):
        bar = "#" * int(p * 40)
        print(f"  {cls:8s} {p*100:5.1f}%  {bar}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict.py path/to/image.png")
        sys.exit(1)
    predict(sys.argv[1])
