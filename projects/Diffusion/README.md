# Nano Stable Diffusion Project

This project adds a tiny, educational, Stable Diffusion-inspired pipeline to NanoTorch.

It is not a reproduction of the original Stable Diffusion model. Instead, it keeps the same broad idea:

- learn a small latent space with an autoencoder
- diffuse inside that latent space instead of raw pixels
- condition denoising on a text-like prompt signal
- decode the final latent back into an image

## What it does

`nano_stable_diffusion.py` trains two small models on a procedurally generated toy dataset of prompt-image pairs:

1. `TinyAutoencoder`
   Compresses a `16x16` grayscale image into a small latent vector.

2. `TinyDenoiser`
   Learns to predict diffusion noise from:
   - noisy latent
   - timestep embedding
   - prompt embedding

The prompts are simple class-like text labels such as:

- `vertical line`
- `horizontal line`
- `cross`
- `x shape`
- `square`
- `diamond`
- `plus`
- `frame`

## Why this is "Stable Diffusion-inspired"

The part that matches the Stable Diffusion idea is the latent diffusion setup:

- image -> latent
- add noise in latent space
- denoise conditioned on prompt
- latent -> image

The parts that are intentionally much smaller and simpler:

- no CLIP text encoder
- no U-Net
- no VAE with KL term
- no large-scale dataset
- no photorealistic output

This is a teaching project, not a production generator.

## Run

```powershell
python projects/Diffusion/nano_stable_diffusion.py
```

Faster smoke test:

```powershell
python projects/Diffusion/nano_stable_diffusion.py --autoencoder-steps 20 --diffusion-steps 40 --samples-per-prompt 2
```

## Outputs

Artifacts are written to `projects/Diffusion/artifacts/`:

- `training_history.json`
- `generated_samples.png`

## Notes

- The dataset is generated on the fly, so no external download is required.
- The implementation depends only on `numpy` and `matplotlib`, plus the local NanoTorch package.
