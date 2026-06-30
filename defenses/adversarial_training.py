import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from attacks.fgsm import fgsm_attack


def generate_adversarial_dataset(model, x_train, y_train, epsilon=0.2, batch_size=256):
    """Create adversarial versions of the training set using CURRENT model weights."""
    adv_list = []
    total = len(x_train)
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        adv_batch = fgsm_attack(model, x_train[start:end], y_train[start:end], epsilon)
        adv_list.append(adv_batch)
    return np.concatenate(adv_list, axis=0)


def adversarial_train(model, x_train, y_train, epsilon=0.2,
                      epochs=10, batch_size=128, validation_split=0.1,
                      x_val=None, y_val=None, verbose=1):
    """
    Retrain the model on a 50/50 mix of clean and FRESH adversarial examples,
    regenerated every epoch from the model's current weights.

    Args:
        model:             Compiled Keras model (ideally already trained on
                            clean data first, so it has a reasonable starting
                            point before robustness fine-tuning begins).
        x_train, y_train:  Clean training data (one-hot labels).
        epsilon:           FGSM attack strength used to generate adversarial
                            training examples.
        epochs:            Number of adversarial-training epochs.
        batch_size:        Mini-batch size for model.fit per epoch.
        validation_split:  Used only if x_val/y_val are not provided.
        x_val, y_val:       Optional explicit validation set (recommended --
                            evaluate on CLEAN validation data so you can see
                            if clean accuracy is being preserved).
        verbose:           Keras fit verbosity.

    Returns:
        history: dict with keys 'loss', 'accuracy', 'val_loss', 'val_accuracy'
                 (one value appended per epoch).
    """
    history = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []}
    n = len(x_train)

    # carve out a fixed clean validation set up front if none given,
    # so robustness/clean-accuracy is judged consistently across epochs
    if x_val is None or y_val is None:
        split = int(n * (1 - validation_split))
        idx = np.random.permutation(n)
        x_train, x_val = x_train[idx[:split]], x_train[idx[split:]]
        y_train, y_val = y_train[idx[:split]], y_train[idx[split:]]
        n = len(x_train)

    for epoch in range(1, epochs + 1):
        print(f"[Adversarial Training] Epoch {epoch}/{epochs}: "
              f"generating fresh adversarial examples (eps={epsilon}) ...")

        # regenerate adversarial examples using the model's CURRENT weights
        adv_images = generate_adversarial_dataset(model, x_train, y_train,
                                                   epsilon=epsilon, batch_size=256)

        x_aug = np.concatenate([x_train, adv_images], axis=0)
        y_aug = np.concatenate([y_train, y_train], axis=0)
        idx = np.random.permutation(len(x_aug))
        x_aug, y_aug = x_aug[idx], y_aug[idx]

        # one epoch of training on this epoch's fresh clean+adversarial mix
        result = model.fit(x_aug, y_aug, epochs=1, batch_size=batch_size,
                           verbose=verbose)

        # evaluate on a held-out CLEAN validation set -- this is what tells
        # you whether the model is staying accurate on normal data while
        # gaining robustness, not just memorizing one batch of noise
        val_loss, val_acc = model.evaluate(x_val, y_val, verbose=0)

        history["loss"].append(result.history["loss"][0])
        history["accuracy"].append(result.history["accuracy"][0])
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)

        print(f"  -> train_acc={history['accuracy'][-1]*100:.2f}%  "
              f"clean_val_acc={val_acc*100:.2f}%")

    print("[Adversarial Training] Done.")
    return history