# FGSM Adversarial Attack & Defense on MNIST

## Overview
This project demonstrates how a trained CNN can be fooled by the **Fast Gradient Sign Method (FGSM)** attack and how **Adversarial Training** can improve robustness.

## Project Structure
```
FGSM_Project/
├── dataset/              ← digit dataset (.npz)
├── models/               ← saved Keras models (.h5)
│   ├── cnn_model.h5
│   └── defended_cnn.h5
├── attacks/
│   └── fgsm.py           ← FGSM attack implementation
├── defenses/
│   └── adversarial_training.py  ← defense via augmented retraining
├── notebooks/
│   └── experiment.ipynb  ← interactive walkthrough
├── results/
│   ├── graphs/           ← accuracy curves, comparison plots
│   └── images/           ← adversarial sample visualisations
└── main.py               ← full end-to-end pipeline
```


## Methodology

Load Data → Train CNN → FGSM Attack → Adversarial Training → Compare Results

A standard CNN is trained on digit images, attacked using FGSM at increasing strengths, then a second copy of the model is retrained with adversarial examples mixed into every epoch to build robustness. Both models are then attacked under identical conditions and compared.


## Quick Start
```bash
pip install tensorflow numpy matplotlib scikit-learn jupyter

# Run the full pipeline
python main.py

# Or open the notebook
jupyter notebook notebooks/experiment.ipynb
```

## How FGSM Works
```
x_adv = x + ε · sign(∇_x J(θ, x, y))
```
A single gradient step in the direction that maximises the loss produces an image that looks identical to humans but fools the model.

## Results Summary
|| epsilon | Standard CNN | Defended CNN | Gain |
|---------|-------------|--------------|------|
| 0.00 | 97.78% | 98.89% | +1.11% |
| 0.05 | 89.72% | 94.72% | +5.00% |
| 0.10 | 76.39% | 90.56% | +14.17% |
| 0.15 | 47.22% | 80.28% | +33.06% |
| 0.20 | 17.78% | 68.33% | +50.56% |
| 0.25 | 10.28% | 53.33% | +43.06% |
| 0.30 | 7.50% | 38.06% | +30.56% |
| 0.35 | 3.61% | 23.89% | +20.28% |
| 0.40 | 1.94% | 11.67% | +9.72% |


The defended model outperforms the standard model at **every** epsilon value, with the largest gain (+50.56%) at ε=0.20 — exactly where the standard model has already collapsed but the defended model is still holding together.

## Dataset

Uses scikit-learn's `load_digits()` (1,797 handwritten digit samples, originally 8×8 pixels, upscaled to 28×28 to match standard CNN input size). Used as a lightweight, dependency-free substitute for MNIST.

## Defense: Adversarial Training
Each mini-batch is augmented with its FGSM counterpart at ε=0.15 before the gradient update. This exposes the model to adversarial inputs during training, building immunity.

## References
- Goodfellow et al. (2014) — *Explaining and Harnessing Adversarial Examples*
- Madry et al. (2018) — *Towards Deep Learning Models Resistant to Adversarial Attacks*
