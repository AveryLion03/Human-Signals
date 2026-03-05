"""
train.py
────────
Loads landmarks.csv produced by prepare_data.py, trains a RandomForest +
MLP ensemble classifier, evaluates it, and saves the model to drowsy_model.pkl.

USAGE
-----
    python train.py

OUTPUT
------
    drowsy_model.pkl   ← loaded by face_mesh.py at runtime
    confusion_matrix.png
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection   import train_test_split, cross_val_score
from sklearn.preprocessing     import LabelEncoder, StandardScaler
from sklearn.ensemble          import RandomForestClassifier, VotingClassifier
from sklearn.neural_network    import MLPClassifier
from sklearn.metrics           import (classification_report,
                                       confusion_matrix,
                                       ConfusionMatrixDisplay)
from sklearn.pipeline          import Pipeline

CSV_PATH   = "landmarks.csv"
MODEL_PATH = "drowsy_model.pkl"
RANDOM_STATE = 42

# ── Load data ─────────────────────────────────────────────────────────────────
print(f"Loading {CSV_PATH} ...")
df = pd.read_csv(CSV_PATH)
print(f"  Shape: {df.shape}")
print(f"  Class distribution:\n{df['label'].value_counts()}\n")

X = df.drop(columns=["label"]).values.astype(np.float32)
y = df["label"].values

le = LabelEncoder()
y_enc = le.fit_transform(y)
print(f"Classes: {list(le.classes_)}")

# ── Train / test split ────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, random_state=RANDOM_STATE, stratify=y_enc
)
print(f"\nTrain: {len(X_train)}  |  Test: {len(X_test)}")

# ── Build model pipeline ──────────────────────────────────────────────────────
# Voting ensemble: RandomForest (tree-based, no scaling needed) +
#                  MLP (needs scaling — handled inside pipeline)
rf  = RandomForestClassifier(n_estimators=200, max_depth=None,
                              n_jobs=-1, random_state=RANDOM_STATE)

mlp = Pipeline([
    ("scaler", StandardScaler()),
    ("mlp",    MLPClassifier(hidden_layer_sizes=(256, 128),
                              activation="relu",
                              max_iter=300,
                              random_state=RANDOM_STATE)),
])

ensemble = VotingClassifier(
    estimators=[("rf", rf), ("mlp", mlp)],
    voting="soft",
    n_jobs=-1,
)

# ── Cross-validation (quick sanity check) ─────────────────────────────────────
print("\nRunning 5-fold cross-validation on training set (this may take a minute)...")
cv_scores = cross_val_score(ensemble, X_train, y_train, cv=5, scoring="accuracy", n_jobs=-1)
print(f"  CV accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

# ── Final fit ─────────────────────────────────────────────────────────────────
print("\nFitting final model on full training set...")
ensemble.fit(X_train, y_train)

# ── Evaluate on held-out test set ─────────────────────────────────────────────
y_pred = ensemble.predict(X_test)
print("\nTest set results:")
print(classification_report(y_test, y_pred, target_names=le.classes_))
# Confusion matrix
os.makedirs("Images", exist_ok=True)
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
fig, ax = plt.subplots(figsize=(6, 5))
disp.plot(ax=ax, colorbar=False, cmap="Blues")
plt.title("Confusion Matrix — Drowsiness Classifier")
plt.tight_layout()
plt.savefig(os.path.join("Images", "confusion_matrix.png"), dpi=120)
print("Confusion matrix saved → Images/confusion_matrix.png")
plt.close()

# ── Save model + encoder ──────────────────────────────────────────────────────
payload = {"model": ensemble, "label_encoder": le}
with open(MODEL_PATH, "wb") as f:
    pickle.dump(payload, f)
print(f"\nModel saved → {MODEL_PATH}")
print("Ready!  Run face_recog.py to use it with your webcam.")
