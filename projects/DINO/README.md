# Nano DINO Project

This project adds a NanoTorch-native, self-supervised DINO-style training example.

It is intentionally small and educational. The goal is to demonstrate the key DINO ingredients inside this repository:

- student and teacher networks
- teacher exponential moving average updates
- multi-view augmentations
- centering
- temperature-scaled distillation loss

## Scope

This is not the object-detection model called DINO, and it is not a reproduction of large-scale DINOv2 training.

It is a small self-supervised representation-learning project for the local `NanoDigits` dataset using NanoTorch modules.

## What it trains on

- dataset: `datasets/nanodigits`
- input: `8x8` grayscale digit images
- supervision during training: none

Labels are only used after training for a simple representation-quality evaluation.

## Run

```powershell
python projects/DINO/nano_dino.py
```

Short smoke run:

```powershell
python projects/DINO/nano_dino.py --epochs 5 --batch-size 32
```

## Outputs

Artifacts are written to `projects/DINO/artifacts/`:

- `training_history.json`
- `training_curves.png`

## Design

The project uses:

- a small MLP backbone over flattened `8x8` inputs
- a projection head to DINO logits
- two augmented views per image
- teacher probabilities from the opposite view
- EMA teacher parameter updates

The evaluation is a simple nearest-class-centroid probe on the learned backbone embeddings.
