"""
FGSM (Fast Gradient Sign Method) Attack
Paper: Goodfellow et al., "Explaining and Harnessing Adversarial Examples" (2014)

The attack perturbs input x as:
    x_adv = x + epsilon * sign(∇_x J(θ, x, y))
"""

import numpy as np
import tensorflow as tf


def fgsm_attack(model, images, labels, epsilon):
    """
    Generate adversarial examples using FGSM.

    Args:
        model:   Trained Keras model.
        images:  Clean input images, shape (N, H, W, C), float32 in [0, 1].
        labels:  One-hot encoded true labels, shape (N, num_classes).
        epsilon: Perturbation magnitude (attack strength).

    Returns:
        adv_images: Adversarially perturbed images clipped to [0, 1].
    """
    images = tf.cast(images, tf.float32)
    labels = tf.cast(labels, tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(images)
        predictions = model(images, training=False)
        loss = tf.keras.losses.categorical_crossentropy(labels, predictions)

    gradients = tape.gradient(loss, images)
    perturbation = epsilon * tf.sign(gradients)
    adv_images = images + perturbation
    adv_images = tf.clip_by_value(adv_images, 0.0, 1.0)
    return adv_images.numpy()


def evaluate_under_attack(model, x_test, y_test, epsilon, batch_size=256):
    """Single-epsilon evaluation; returns a float accuracy."""
    correct = 0
    total   = len(x_test)
    for start in range(0, total, batch_size):
        end     = min(start + batch_size, total)
        adv     = fgsm_attack(model, x_test[start:end], y_test[start:end], epsilon)
        preds   = model.predict(adv, verbose=0)
        correct += int((preds.argmax(1) == y_test[start:end].argmax(1)).sum())
    return correct / total


def evaluate_attack(model, images, labels, epsilons):
    """
    Evaluate model accuracy under FGSM across a range of epsilon values.

    Returns:
        accuracies: List of accuracies corresponding to each epsilon.
    """
    accuracies = []
    for eps in epsilons:
        adv_images = fgsm_attack(model, images, labels, eps)
        loss, acc = model.evaluate(adv_images, labels, verbose=0)
        accuracies.append(acc)
        print(f"  eps={eps:.3f}  ->  accuracy={acc*100:.2f}%")
    return accuracies
