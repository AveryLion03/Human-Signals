"""
train.py
────────
Loads landmarks.csv produced by prepare_data.py.

Models trained & benchmarked:
  1. RandomForest + MLP Ensemble  (original)
  2. SVM                          (new)
  3. KNN                          (new)
  4. 1D-CNN via Keras/TensorFlow  (new)

Features: 478 landmark (x,y,z) coords + ear_left, ear_right, ear_avg,
          mar (mouth aspect ratio), head_pose_x, head_pose_y, head_pose_z  ← add these in prepare_data.py

USAGE
-----
    pip install tensorflow scikit-learn pandas matplotlib
    python train.py

OUTPUT
------
    drowsy_model.pkl           (best sklearn model)
    drowsy_cnn_model.keras     (1D-CNN)
    Images/confusion_matrix_<model>.png
    Images/model_comparison.png
    Images/roc_curves.png
"""

import os
import pickle
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── Sklearn ────────────────────────────────────────────────────────────────────
from sklearn.model_selection    import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing      import LabelEncoder, StandardScaler
from sklearn.ensemble           import RandomForestClassifier, VotingClassifier
from sklearn.neural_network     import MLPClassifier
from sklearn.svm                import SVC
from sklearn.neighbors          import KNeighborsClassifier
from sklearn.metrics            import (classification_report,
                                        confusion_matrix,
                                        ConfusionMatrixDisplay,
                                        roc_curve, auc,
                                        f1_score, accuracy_score)
from sklearn.pipeline           import Pipeline
from sklearn.decomposition      import PCA

# ── TensorFlow / Keras ────────────────────────────────────────────────────────
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, callbacks
    TF_AVAILABLE = True
    print(f"TensorFlow {tf.__version__} detected ✓")
except ImportError:
    TF_AVAILABLE = False
    print("[WARN] TensorFlow not found — CNN model will be skipped.")
    print("       Install with: pip install tensorflow")

# ─────────────────────────────────────────────────────────────────────────────
CSV_PATH     = "landmarks.csv"
MODEL_PATH   = "drowsy_model.pkl"
CNN_PATH     = "drowsy_cnn_model.keras"
RANDOM_STATE = 42
os.makedirs("Images", exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"Loading {CSV_PATH} ...")
df = pd.read_csv(CSV_PATH)
print(f"  Shape: {df.shape}")
print(f"  Class distribution:\n{df['label'].value_counts()}\n")

# ── Feature engineering ───────────────────────────────────────────────────────
# EAR columns (required)
ear_cols = ["ear_left", "ear_right", "ear_avg"]
for col in ear_cols:
    if col not in df.columns:
        print(f"[WARN] '{col}' missing — re-run prepare_data.py")

# Optional enhanced features — add these in prepare_data.py for best results
optional_cols = ["mar", "head_pose_x", "head_pose_y", "head_pose_z",
                 "blink_rate", "perclos"]
present_optional = [c for c in optional_cols if c in df.columns]
if present_optional:
    print(f"  Optional features found: {present_optional}")
else:
    print("  [TIP] Add MAR, head pose, PERCLOS to prepare_data.py for higher accuracy.\n"
          "        See comments at bottom of this file.")

X = df.drop(columns=["label"]).values.astype(np.float32)
y = df["label"].values

le = LabelEncoder()
y_enc = le.fit_transform(y)
n_classes = len(le.classes_)
print(f"Classes: {list(le.classes_)}  (encoded: {list(range(n_classes))})")
print(f"Feature dimensionality: {X.shape[1]}")

# ── Train / test split ────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, random_state=RANDOM_STATE, stratify=y_enc
)
print(f"\nTrain: {len(X_train)}  |  Test: {len(X_test)}")

# ══════════════════════════════════════════════════════════════════════════════
# 2. DEFINE MODELS
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("Building model pipelines...")

# ── 2a. Original Ensemble (RF + MLP) ─────────────────────────────────────────
rf = RandomForestClassifier(
    n_estimators=300,           # increased from 200
    class_weight="balanced",
    max_depth=None,
    min_samples_leaf=2,
    n_jobs=-1,
    random_state=RANDOM_STATE,
)

mlp = Pipeline([
    ("scaler", StandardScaler()),
    ("mlp", MLPClassifier(
        hidden_layer_sizes=(512, 256, 128),   # deeper than original
        activation="relu",
        max_iter=500,
        early_stopping=True,      # NEW: prevents overfitting
        validation_fraction=0.1,
        n_iter_no_change=20,
        random_state=RANDOM_STATE,
    )),
])

ensemble = VotingClassifier(
    estimators=[("rf", rf), ("mlp", mlp)],
    voting="soft",
    n_jobs=-1,
)

# ── 2b. SVM ───────────────────────────────────────────────────────────────────
# PCA first — SVM is O(n²~n³), dimensionality reduction is essential here
# with 1400+ features. 95% variance retained.
svm_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("pca",    PCA(n_components=0.95, random_state=RANDOM_STATE)),  # retain 95% variance
    ("svm",    SVC(
        kernel="rbf",
        C=10.0,           # regularization — tune with GridSearchCV if needed
        gamma="scale",
        class_weight="balanced",
        probability=True, # needed for ROC curves and soft voting
        random_state=RANDOM_STATE,
    )),
])

# ── 2c. KNN ───────────────────────────────────────────────────────────────────
# PCA + scaling essential for KNN — it's distance-based, so high-dim is problematic
knn_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("pca",    PCA(n_components=100, random_state=RANDOM_STATE)),  # fixed 100 components
    ("knn",    KNeighborsClassifier(
        n_neighbors=7,       # odd number avoids ties
        weights="distance",  # closer neighbors weighted more
        metric="euclidean",
        n_jobs=-1,
    )),
])

sklearn_models = {
    "RF+MLP Ensemble": ensemble,
    "SVM (RBF)":       svm_pipeline,
    "KNN (k=7)":       knn_pipeline,
}

# ══════════════════════════════════════════════════════════════════════════════
# 3. CROSS-VALIDATION BENCHMARK (sklearn models)
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("Running 5-fold stratified cross-validation...\n")

cv_results = {}
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

for name, model in sklearn_models.items():
    t0 = time.time()
    scores = cross_val_score(model, X_train, y_train,
                             cv=skf, scoring="f1_weighted", n_jobs=-1)
    elapsed = time.time() - t0
    cv_results[name] = scores
    print(f"  {name:<22} F1: {scores.mean():.3f} ± {scores.std():.3f}  ({elapsed:.1f}s)")

# ══════════════════════════════════════════════════════════════════════════════
# 4. FIT ALL SKLEARN MODELS ON FULL TRAINING SET
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("Fitting models on full training set...\n")

fitted_models = {}
test_results  = {}

for name, model in sklearn_models.items():
    print(f"  Fitting {name}...")
    t0 = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - t0

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, average="weighted")

    fitted_models[name] = model
    test_results[name]  = {
        "acc": acc, "f1": f1,
        "y_pred": y_pred, "y_prob": y_prob,
        "train_time": elapsed
    }

    print(f"    Acc: {acc:.3f}  |  F1: {f1:.3f}  |  Time: {elapsed:.1f}s")
    print(classification_report(y_test, y_pred,
                                target_names=le.classes_, zero_division=0))

# ══════════════════════════════════════════════════════════════════════════════
# 5. 1D-CNN (TensorFlow/Keras)
# ══════════════════════════════════════════════════════════════════════════════
if TF_AVAILABLE:
    print(f"\n{'='*60}")
    print("Training 1D-CNN...\n")

    # Scale data
    cnn_scaler  = StandardScaler()
    X_train_cnn = cnn_scaler.fit_transform(X_train)
    X_test_cnn  = cnn_scaler.transform(X_test)

    # Reshape to (samples, timesteps, channels) — treat features as a 1D sequence
    X_train_cnn = X_train_cnn.reshape(X_train_cnn.shape[0], X_train_cnn.shape[1], 1)
    X_test_cnn  = X_test_cnn.reshape(X_test_cnn.shape[0], X_test_cnn.shape[1], 1)

    n_features = X_train_cnn.shape[1]

    # ── Architecture ──────────────────────────────────────────────────────────
    # Treat the feature vector as a 1D signal.
    # Conv layers learn local feature relationships (e.g., eye landmark clusters).
    inputs = keras.Input(shape=(n_features, 1), name="landmarks")

    x = layers.Conv1D(64, kernel_size=7, activation="relu", padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv1D(128, kernel_size=5, activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv1D(256, kernel_size=3, activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling1D()(x)  # better than Flatten for generalization

    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(64, activation="relu")(x)

    # Output: sigmoid for binary, softmax for multi-class
    if n_classes == 2:
        outputs = layers.Dense(1, activation="sigmoid")(x)
        loss    = "binary_crossentropy"
    else:
        outputs = layers.Dense(n_classes, activation="softmax")(x)
        loss    = "sparse_categorical_crossentropy"

    cnn_model = keras.Model(inputs, outputs, name="Drowsiness_1DCNN")
    cnn_model.summary()

    cnn_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=loss,
        metrics=["accuracy"]
    )

    cnn_callbacks = [
        callbacks.EarlyStopping(patience=15, restore_best_weights=True,
                                monitor="val_accuracy"),
        callbacks.ReduceLROnPlateau(factor=0.5, patience=7, min_lr=1e-6,
                                    monitor="val_loss"),
        callbacks.ModelCheckpoint(CNN_PATH, save_best_only=True,
                                  monitor="val_accuracy"),
    ]

    y_train_cnn = y_train if n_classes > 2 else y_train.astype(np.float32)
    y_test_cnn  = y_test  if n_classes > 2 else y_test.astype(np.float32)

    history = cnn_model.fit(
        X_train_cnn, y_train_cnn,
        validation_split=0.15,
        epochs=100,
        batch_size=32,
        callbacks=cnn_callbacks,
        verbose=1,
    )

    # ── CNN Evaluate ──────────────────────────────────────────────────────────
    cnn_pred_raw = cnn_model.predict(X_test_cnn)
    if n_classes == 2:
        y_prob_cnn  = np.hstack([1 - cnn_pred_raw, cnn_pred_raw])
        y_pred_cnn  = (cnn_pred_raw > 0.5).astype(int).flatten()
    else:
        y_prob_cnn  = cnn_pred_raw
        y_pred_cnn  = np.argmax(cnn_pred_raw, axis=1)

    acc_cnn = accuracy_score(y_test, y_pred_cnn)
    f1_cnn  = f1_score(y_test, y_pred_cnn, average="weighted")
    test_results["1D-CNN"] = {
        "acc": acc_cnn, "f1": f1_cnn,
        "y_pred": y_pred_cnn, "y_prob": y_prob_cnn,
        "train_time": None
    }

    print(f"\n1D-CNN  Acc: {acc_cnn:.3f}  |  F1: {f1_cnn:.3f}")
    print(classification_report(y_test, y_pred_cnn,
                                target_names=le.classes_, zero_division=0))

    # ── CNN Training curve ────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history["accuracy"],     label="Train")
    axes[0].plot(history.history["val_accuracy"], label="Val")
    axes[0].set_title("CNN Accuracy"); axes[0].legend()
    axes[1].plot(history.history["loss"],     label="Train")
    axes[1].plot(history.history["val_loss"], label="Val")
    axes[1].set_title("CNN Loss"); axes[1].legend()
    plt.tight_layout()
    plt.savefig("Images/cnn_training_curves.png", dpi=120)
    plt.close()
    print("CNN training curves → Images/cnn_training_curves.png")

    # Save CNN scaler alongside model
    with open("cnn_scaler.pkl", "wb") as f:
        pickle.dump(cnn_scaler, f)

# ══════════════════════════════════════════════════════════════════════════════
# 6. CONFUSION MATRICES (all models)
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("Generating confusion matrices...")

n_models = len(test_results)
fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 5))
if n_models == 1:
    axes = [axes]

for ax, (name, res) in zip(axes, test_results.items()):
    cm   = confusion_matrix(y_test, res["y_pred"])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"{name}\nAcc={res['acc']:.3f}  F1={res['f1']:.3f}", fontsize=10)

plt.suptitle("Confusion Matrices — All Models", fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig("Images/confusion_matrices_all.png", dpi=120, bbox_inches="tight")
plt.close()
print("  Saved → Images/confusion_matrices_all.png")

# ══════════════════════════════════════════════════════════════════════════════
# 7. ROC CURVES (binary classification only)
# ══════════════════════════════════════════════════════════════════════════════
if n_classes == 2:
    print("Generating ROC curves...")
    fig, ax = plt.subplots(figsize=(7, 6))
    colors = ["royalblue", "tomato", "seagreen", "darkorange"]

    for (name, res), color in zip(test_results.items(), colors):
        fpr, tpr, _ = roc_curve(y_test, res["y_prob"][:, 1])
        roc_auc     = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2,
                label=f"{name}  (AUC = {roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — All Models")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig("Images/roc_curves.png", dpi=120)
    plt.close()
    print("  Saved → Images/roc_curves.png")

# ══════════════════════════════════════════════════════════════════════════════
# 8. MODEL COMPARISON BAR CHART
# ══════════════════════════════════════════════════════════════════════════════
print("Generating model comparison chart...")

names   = list(test_results.keys())
accs    = [test_results[n]["acc"] for n in names]
f1s     = [test_results[n]["f1"]  for n in names]
cv_f1s  = [cv_results[n].mean() if n in cv_results else 0 for n in names]

x   = np.arange(len(names))
w   = 0.25
fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(x - w,   accs,   w, label="Test Accuracy", color="steelblue")
ax.bar(x,       f1s,    w, label="Test F1 (weighted)", color="tomato")
ax.bar(x + w,   cv_f1s, w, label="CV F1 (train set)", color="seagreen")

ax.set_xticks(x); ax.set_xticklabels(names, rotation=15, ha="right")
ax.set_ylim(0, 1.05)
ax.set_ylabel("Score"); ax.set_title("Model Comparison")
ax.legend(); ax.grid(axis="y", alpha=0.3)

for i, (a, f, c) in enumerate(zip(accs, f1s, cv_f1s)):
    ax.text(i - w, a + 0.01, f"{a:.3f}", ha="center", fontsize=8)
    ax.text(i,     f + 0.01, f"{f:.3f}", ha="center", fontsize=8)
    if c > 0:
        ax.text(i + w, c + 0.01, f"{c:.3f}", ha="center", fontsize=8)

plt.tight_layout()
plt.savefig("Images/model_comparison.png", dpi=120)
plt.close()
print("  Saved → Images/model_comparison.png")

# ══════════════════════════════════════════════════════════════════════════════
# 9. SAVE BEST SKLEARN MODEL
# ══════════════════════════════════════════════════════════════════════════════
best_sklearn_name = max(
    [n for n in sklearn_models], key=lambda n: test_results[n]["f1"]
)
best_model = fitted_models[best_sklearn_name]
print(f"\n{'='*60}")
print(f"Best sklearn model: {best_sklearn_name}  (F1={test_results[best_sklearn_name]['f1']:.3f})")

payload = {
    "model":         best_model,
    "label_encoder": le,
    "best_name":     best_sklearn_name,
    "all_results":   {n: {"acc": r["acc"], "f1": r["f1"]}
                      for n, r in test_results.items()},
}
with open(MODEL_PATH, "wb") as f:
    pickle.dump(payload, f)
print(f"Best model saved → {MODEL_PATH}")

if TF_AVAILABLE:
    print(f"CNN model saved  → {CNN_PATH}")

print("\n✓ Done! Run face_mesh.py to use the model with your webcam.")

# ══════════════════════════════════════════════════════════════════════════════
# APPENDIX: How to add MAR + Head Pose to prepare_data.py
# ══════════════════════════════════════════════════════════════════════════════
"""
── MOUTH ASPECT RATIO (MAR) ──────────────────────────────────────────────────
Add to prepare_data.py after computing EAR:

MOUTH_TOP    = [13, 312, 311, 310]
MOUTH_BOTTOM = [14, 82, 81, 80]
MOUTH_LEFT   = 61
MOUTH_RIGHT  = 291

def mouth_aspect_ratio(lm, w, h):
    top    = np.mean([[lm[i].x * w, lm[i].y * h] for i in MOUTH_TOP],    axis=0)
    bottom = np.mean([[lm[i].x * w, lm[i].y * h] for i in MOUTH_BOTTOM], axis=0)
    left   = np.array([lm[MOUTH_LEFT].x  * w, lm[MOUTH_LEFT].y  * h])
    right  = np.array([lm[MOUTH_RIGHT].x * w, lm[MOUTH_RIGHT].y * h])
    vertical   = np.linalg.norm(top - bottom)
    horizontal = np.linalg.norm(left - right)
    return vertical / (horizontal + 1e-6)

── HEAD POSE (via solvePnP) ──────────────────────────────────────────────────
import cv2

MODEL_POINTS = np.array([
    [0.0, 0.0, 0.0],          # Nose tip (1)
    [0.0, -330.0, -65.0],     # Chin (152)
    [-225.0, 170.0, -135.0],  # Left eye corner (263)
    [225.0, 170.0, -135.0],   # Right eye corner (33)
    [-150.0, -150.0, -125.0], # Left mouth corner (287)
    [150.0, -150.0, -125.0],  # Right mouth corner (57)
])
MP_INDICES = [1, 152, 263, 33, 287, 57]

def head_pose(lm, w, h):
    img_pts = np.array(
        [[lm[i].x * w, lm[i].y * h] for i in MP_INDICES], dtype=np.float64
    )
    focal   = w
    cam_mat = np.array([[focal, 0, w/2], [0, focal, h/2], [0, 0, 1]], dtype=np.float64)
    _, rvec, _ = cv2.solvePnP(MODEL_POINTS, img_pts, cam_mat,
                               np.zeros((4,1)), flags=cv2.SOLVEPNP_ITERATIVE)
    rot, _ = cv2.Rodrigues(rvec)
    pitch  = np.degrees(np.arcsin(-rot[2, 0]))
    yaw    = np.degrees(np.arctan2(rot[2, 1], rot[2, 2]))
    roll   = np.degrees(np.arctan2(rot[1, 0], rot[0, 0]))
    return pitch, yaw, roll
"""