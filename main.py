"""
FGSM Adversarial Attack & Defense — Main Pipeline
"""

import os, sys
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import confusion_matrix

sys.path.insert(0, os.path.dirname(__file__))
from attacks.fgsm import fgsm_attack, evaluate_attack
from defenses.adversarial_training import adversarial_train

np.random.seed(42); tf.random.set_seed(42)

RESULTS_GRAPHS = "results/graphs"
RESULTS_IMAGES = "results/images"
MODELS_DIR     = "models"

# ── 1. DATA ───────────────────────────────────────────────────────────────────
print("\n[1/6] Loading dataset …")
data = np.load("dataset/data.npz")
x_train, y_train_raw = data["x_train"], data["y_train"]
x_test,  y_test_raw  = data["x_test"],  data["y_test"]
y_train = tf.keras.utils.to_categorical(y_train_raw, 10)
y_test  = tf.keras.utils.to_categorical(y_test_raw,  10)
print(f"   Train: {x_train.shape}  Test: {x_test.shape}")

# ── 2. CNN ────────────────────────────────────────────────────────────────────
def build_cnn():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(32, 3, activation="relu", padding="same"),
        tf.keras.layers.Conv2D(32, 3, activation="relu", padding="same"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same"),
        tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(10, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model

MODEL_PATH = os.path.join(MODELS_DIR, "cnn_model.h5")
print("\n[2/6] Training CNN …")
model = build_cnn()
cb = [tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)]
history = model.fit(x_train, y_train, epochs=30, batch_size=64,
                    validation_split=0.15, callbacks=cb, verbose=0)
model.save(MODEL_PATH)
_, clean_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"   Clean test accuracy: {clean_acc*100:.2f}%")

# Training curve
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(history.history["accuracy"],     label="Train")
axes[0].plot(history.history["val_accuracy"], label="Validation")
axes[0].set_title("Model Accuracy", fontsize=13); axes[0].legend(); axes[0].grid(alpha=.3)
axes[1].plot(history.history["loss"],     label="Train")
axes[1].plot(history.history["val_loss"], label="Validation")
axes[1].set_title("Model Loss", fontsize=13); axes[1].legend(); axes[1].grid(alpha=.3)
plt.suptitle("CNN Training Curve", fontsize=15, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_GRAPHS, "training_curve.png"), dpi=150)
plt.close()
print("   Saved training_curve.png")

# ── 3. FGSM ATTACK ─────────────────────────────────────────────────────────
print("\n[3/6] Running FGSM attack …")
epsilons = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
clean_accs = evaluate_attack(model, x_test, y_test, epsilons)

plt.figure(figsize=(8, 5))
plt.plot([e*100 for e in epsilons], [a*100 for a in clean_accs],
         "o-", color="#E74C3C", lw=2.5, ms=8, label="Standard CNN", zorder=3)
plt.fill_between([e*100 for e in epsilons], [a*100 for a in clean_accs],
                 alpha=0.1, color="#E74C3C")
plt.axhline(10, color="gray", linestyle="--", alpha=0.5, label="Random guess (10%)")
plt.xlabel("Perturbation Strength ε (×100)", fontsize=12)
plt.ylabel("Accuracy (%)", fontsize=12)
plt.title("FGSM Attack: How Accuracy Degrades with ε", fontsize=13, fontweight="bold")
plt.grid(alpha=.3); plt.legend(fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_GRAPHS, "accuracy_vs_epsilon.png"), dpi=150)
plt.close()
print("   Saved accuracy_vs_epsilon.png")

# ── 4. SAMPLE IMAGES ─────────────────────────────────────────────────────────
print("\n[4/6] Generating sample adversarial images …")
sample_idx = [np.where(y_test_raw == d)[0][0] for d in range(10)]
x_sample = x_test[sample_idx]
y_sample = y_test[sample_idx]

for eps_val in [0.10, 0.25]:
    adv = fgsm_attack(model, x_sample, y_sample, eps_val)
    orig_preds = np.argmax(model.predict(x_sample, verbose=0), axis=1)
    adv_preds  = np.argmax(model.predict(adv,       verbose=0), axis=1)

    fig, axes = plt.subplots(2, 10, figsize=(20, 5))
    for i in range(10):
        axes[0, i].imshow(x_sample[i, :, :, 0], cmap="gray")
        axes[0, i].set_title(f"True:{y_test_raw[sample_idx[i]]}\nPred:{orig_preds[i]}", fontsize=7)
        axes[0, i].axis("off")
        axes[1, i].imshow(adv[i, :, :, 0], cmap="gray")
        color = "#E74C3C" if adv_preds[i] != y_test_raw[sample_idx[i]] else "#27AE60"
        axes[1, i].set_title(f"Adv:{adv_preds[i]}", fontsize=7, color=color)
        axes[1, i].axis("off")

    fig.text(0.01, 0.75, "Original", va="center", fontsize=10, fontweight="bold")
    fig.text(0.01, 0.25, "Adversarial", va="center", fontsize=10, fontweight="bold", color="#E74C3C")
    plt.suptitle(f"FGSM Attack Samples  (ε = {eps_val})  — Red=Fooled, Green=Correct", fontsize=11)
    fname = f"adversarial_samples_eps{int(eps_val*100):02d}.png"
    plt.savefig(os.path.join(RESULTS_IMAGES, fname), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   Saved {fname}")

# Perturbation map
adv_mid = fgsm_attack(model, x_sample, y_sample, 0.15)
fig, axes = plt.subplots(3, 10, figsize=(20, 7))
row_labels = ["Original", "Noise ×10", "Adversarial"]
for i in range(10):
    axes[0, i].imshow(x_sample[i, :, :, 0],                    cmap="gray")
    axes[1, i].imshow(np.abs(adv_mid[i]-x_sample[i])[:,:,0]*10,cmap="hot")
    axes[2, i].imshow(adv_mid[i, :, :, 0],                     cmap="gray")
    for r in range(3): axes[r, i].axis("off")
for r, lbl in enumerate(row_labels):
    axes[r, 0].set_ylabel(lbl, fontsize=9, fontweight="bold")
plt.suptitle("Perturbation Visualisation  (ε=0.15 — noise magnified ×10)", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_IMAGES, "perturbation_visualisation.png"), dpi=150)
plt.close()
print("   Saved perturbation_visualisation.png")

# ── 5. ADVERSARIAL TRAINING ──────────────────────────────────────────────────
print("\n[5/6] Adversarial training (defense) …")
DEFENDED_PATH = os.path.join(MODELS_DIR, "defended_cnn.h5")
defended_model = build_cnn()
defended_model.fit(x_train, y_train, epochs=5, batch_size=64,
                   validation_split=0.15, verbose=0)
adv_hist = adversarial_train(defended_model, x_train, y_train,
                              epsilon=0.15, epochs=8, batch_size=64)
defended_model.save(DEFENDED_PATH)

plt.figure(figsize=(8, 5))
plt.plot(adv_hist["accuracy"],     "o-", color="#3498DB", lw=2, ms=6, label="Train acc")
plt.plot(adv_hist["val_accuracy"], "s-", color="#27AE60", lw=2, ms=6, label="Val acc")
plt.xlabel("Epoch"); plt.ylabel("Accuracy")
plt.title("Adversarial Training Progress", fontsize=13, fontweight="bold")
plt.legend(); plt.grid(alpha=.3)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_GRAPHS, "adversarial_training_curve.png"), dpi=150)
plt.close()
print("   Saved adversarial_training_curve.png")

# ── 6. COMPARISON ────────────────────────────────────────────────────────────
print("\n[6/6] Comparing models …")
defended_accs = evaluate_attack(defended_model, x_test, y_test, epsilons)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Accuracy vs epsilon
axes[0].plot([e*100 for e in epsilons], [a*100 for a in clean_accs],
             "o-", color="#E74C3C", lw=2.5, ms=8, label="Standard CNN")
axes[0].plot([e*100 for e in epsilons], [a*100 for a in defended_accs],
             "s-", color="#27AE60", lw=2.5, ms=8, label="Adversarially Trained CNN")
axes[0].fill_between([e*100 for e in epsilons],
                     [a*100 for a in clean_accs], [a*100 for a in defended_accs],
                     alpha=0.12, color="#27AE60", label="Robustness gain")
axes[0].axhline(10, color="gray", linestyle="--", alpha=0.4, label="Random (10%)")
axes[0].set_xlabel("ε (×100)", fontsize=11); axes[0].set_ylabel("Accuracy (%)", fontsize=11)
axes[0].set_title("Accuracy vs Epsilon", fontsize=12, fontweight="bold")
axes[0].legend(fontsize=9); axes[0].grid(alpha=.3)

# Bar chart at selected epsilons
sel = [0, 0.15, 0.30]
sel_idx = [epsilons.index(e) for e in sel]
x_pos = np.arange(len(sel)); w = 0.35
axes[1].bar(x_pos - w/2, [clean_accs[i]*100 for i in sel_idx],  w,
            color="#E74C3C", label="Standard CNN",              alpha=0.85)
axes[1].bar(x_pos + w/2, [defended_accs[i]*100 for i in sel_idx], w,
            color="#27AE60", label="Adversarially Trained CNN", alpha=0.85)
axes[1].set_xticks(x_pos)
axes[1].set_xticklabels([f"ε={e}" for e in sel], fontsize=11)
axes[1].set_ylabel("Accuracy (%)", fontsize=11)
axes[1].set_title("Accuracy at Key Epsilon Values", fontsize=12, fontweight="bold")
axes[1].legend(fontsize=9); axes[1].grid(axis="y", alpha=.3)

plt.suptitle("Standard CNN vs Adversarially Trained CNN", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_GRAPHS, "comparison_accuracy.png"), dpi=150)
plt.close()
print("   Saved comparison_accuracy.png")

# Confusion matrix
adv20 = fgsm_attack(model, x_test, y_test, 0.20)
preds = np.argmax(model.predict(adv20, verbose=0), axis=1)
cm = confusion_matrix(y_test_raw, preds)
fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(cm, cmap="Blues")
plt.colorbar(im)
ax.set_xticks(range(10)); ax.set_yticks(range(10))
ax.set_xlabel("Predicted Label", fontsize=11); ax.set_ylabel("True Label", fontsize=11)
ax.set_title("Confusion Matrix — Standard CNN under FGSM (ε=0.20)", fontsize=12, fontweight="bold")
thresh = cm.max() / 2
for r in range(10):
    for c in range(10):
        ax.text(c, r, str(cm[r, c]), ha="center", va="center",
                fontsize=8, color="white" if cm[r, c] > thresh else "black")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_GRAPHS, "confusion_matrix_attacked.png"), dpi=150)
plt.close()
print("   Saved confusion_matrix_attacked.png")

# Summary table
print("\n" + "="*55)
print(f"{'epsilon':>10}  {'Standard':>12}  {'Defended':>12}  {'Gain':>8}")
print("-"*55)
for i, e in enumerate(epsilons):
    gain = (defended_accs[i] - clean_accs[i]) * 100
    print(f"  {e:8.2f}  {clean_accs[i]*100:11.2f}%  {defended_accs[i]*100:11.2f}%  {gain:+7.2f}%")
print("="*55)
print(f"\n✓ All outputs saved to results/")
