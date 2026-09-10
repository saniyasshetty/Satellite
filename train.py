"""
train.py
--------
Loads data/train and data/test (created by data_generator.py), extracts
features (features.py), trains a classifier, evaluates it, and saves the
trained model to models/poverty_model.joblib.

Run:
    python train.py
"""

import os
import glob
import time
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from features import extract_features_batch, FEATURE_NAMES

CLASSES = ["low", "medium", "high"]
DATA_ROOT = "data"
MODEL_PATH = "models/poverty_model.joblib"
PLOT_PATH = "models/confusion_matrix.png"
IMPORTANCE_PLOT_PATH = "models/feature_importance.png"


def load_split(split):
    paths, labels = [], []
    for cls in CLASSES:
        cls_paths = sorted(glob.glob(os.path.join(DATA_ROOT, split, cls, "*.png")))
        paths.extend(cls_paths)
        labels.extend([cls] * len(cls_paths))
    return paths, labels


def main():
    os.makedirs("models", exist_ok=True)

    print("Loading training data...")
    train_paths, train_labels = load_split("train")
    test_paths, test_labels = load_split("test")
    print(f"  train: {len(train_paths)} images | test: {len(test_paths)} images")

    print("Extracting features (vegetation, built-up density, edges, texture)...")
    t0 = time.time()
    X_train = extract_features_batch(train_paths)
    X_test = extract_features_batch(test_paths)
    print(f"  done in {time.time()-t0:.1f}s | feature vector length = {X_train.shape[1]}")

    y_train = np.array(train_labels)
    y_test = np.array(test_labels)

    print("Training RandomForest classifier...")
    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    print("Evaluating on held-out test set...")
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nTest Accuracy: {acc*100:.2f}%\n")
    print(classification_report(y_test, y_pred, target_names=CLASSES))

    # Confusion matrix plot
    cm = confusion_matrix(y_test, y_pred, labels=CLASSES)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(CLASSES)))
    ax.set_yticks(range(len(CLASSES)))
    ax.set_xticklabels(CLASSES)
    ax.set_yticklabels(CLASSES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix (Acc: {acc*100:.1f}%)")
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                     color="white" if cm[i, j] > cm.max()/2 else "black")
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=120)
    print(f"Saved confusion matrix -> {PLOT_PATH}")

    # Feature importance plot
    importances = clf.feature_importances_
    order = np.argsort(importances)
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.barh(np.array(FEATURE_NAMES)[order], importances[order], color="#2b7a4b")
    ax2.set_title("Feature Importance")
    plt.tight_layout()
    plt.savefig(IMPORTANCE_PLOT_PATH, dpi=120)
    print(f"Saved feature importance plot -> {IMPORTANCE_PLOT_PATH}")

    joblib.dump({"model": clf, "classes": CLASSES, "feature_names": FEATURE_NAMES}, MODEL_PATH)
    print(f"\nSaved trained model -> {MODEL_PATH}")


if __name__ == "__main__":
    main()
