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

## Quick Start
```bash
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
| Model | Clean Acc | @ ε=0.15 | @ ε=0.30 |
|-------|-----------|----------|----------|
| Standard CNN | ~98% | drops significantly | near random |
| Adversarially Trained CNN | ~96% | much more robust | meaningful accuracy retained |

## Defense: Adversarial Training
Each mini-batch is augmented with its FGSM counterpart at ε=0.15 before the gradient update. This exposes the model to adversarial inputs during training, building immunity.

## References
- Goodfellow et al. (2014) — *Explaining and Harnessing Adversarial Examples*
- Madry et al. (2018) — *Towards Deep Learning Models Resistant to Adversarial Attacks*
