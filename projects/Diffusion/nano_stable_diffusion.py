import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import nanotorch as nt
import nanotorch.nn as nn
from nanotorch.optim import Adam


PROMPTS = [
    "vertical line",
    "horizontal line",
    "cross",
    "x shape",
    "square",
    "diamond",
    "plus",
    "frame",
]


def seed_everything(seed):
    np.random.seed(seed)


def clamp01(array):
    return np.clip(array, 0.0, 1.0)


def timestep_embedding(timesteps, dim):
    half = dim // 2
    steps = np.asarray(timesteps, dtype=np.float32).reshape(-1, 1)
    freqs = np.exp(-math.log(10000.0) * np.arange(half, dtype=np.float32) / max(half - 1, 1))
    angles = steps * freqs.reshape(1, -1)
    embedding = np.concatenate([np.sin(angles), np.cos(angles)], axis=1)
    if embedding.shape[1] < dim:
        embedding = np.pad(embedding, ((0, 0), (0, dim - embedding.shape[1])))
    return embedding.astype(np.float32)


def prompt_one_hot(prompt_ids, vocab_size):
    result = np.zeros((len(prompt_ids), vocab_size), dtype=np.float32)
    result[np.arange(len(prompt_ids)), prompt_ids] = 1.0
    return result


def draw_line(canvas, row0, col0, row1, col1, value=1.0):
    rows = np.linspace(row0, row1, 64)
    cols = np.linspace(col0, col1, 64)
    for row, col in zip(rows, cols):
        r = int(np.clip(round(row), 0, canvas.shape[0] - 1))
        c = int(np.clip(round(col), 0, canvas.shape[1] - 1))
        canvas[r, c] = value


def thicken(canvas, radius):
    if radius <= 0:
        return canvas
    out = canvas.copy()
    active = np.argwhere(canvas > 0.0)
    for row, col in active:
        row0 = max(0, row - radius)
        row1 = min(canvas.shape[0], row + radius + 1)
        col0 = max(0, col - radius)
        col1 = min(canvas.shape[1], col + radius + 1)
        out[row0:row1, col0:col1] = np.maximum(out[row0:row1, col0:col1], canvas[row, col])
    return out


def render_prompt(prompt_id, image_size=16, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    canvas = np.zeros((image_size, image_size), dtype=np.float32)
    offset_r = int(rng.integers(-2, 3))
    offset_c = int(rng.integers(-2, 3))
    center = image_size // 2
    thickness = int(rng.integers(1, 3))
    margin = int(rng.integers(2, 4))

    if prompt_id == 0:
        col = np.clip(center + offset_c, margin, image_size - margin - 1)
        draw_line(canvas, margin, col, image_size - margin - 1, col)
    elif prompt_id == 1:
        row = np.clip(center + offset_r, margin, image_size - margin - 1)
        draw_line(canvas, row, margin, row, image_size - margin - 1)
    elif prompt_id == 2:
        row = np.clip(center + offset_r, margin, image_size - margin - 1)
        col = np.clip(center + offset_c, margin, image_size - margin - 1)
        draw_line(canvas, margin, col, image_size - margin - 1, col)
        draw_line(canvas, row, margin, row, image_size - margin - 1)
    elif prompt_id == 3:
        draw_line(canvas, margin, margin, image_size - margin - 1, image_size - margin - 1)
        draw_line(canvas, margin, image_size - margin - 1, image_size - margin - 1, margin)
    elif prompt_id == 4:
        top = np.clip(margin + offset_r, 1, image_size - 6)
        left = np.clip(margin + offset_c, 1, image_size - 6)
        bottom = min(image_size - 2, top + int(rng.integers(6, 10)))
        right = min(image_size - 2, left + int(rng.integers(6, 10)))
        draw_line(canvas, top, left, top, right)
        draw_line(canvas, bottom, left, bottom, right)
        draw_line(canvas, top, left, bottom, left)
        draw_line(canvas, top, right, bottom, right)
    elif prompt_id == 5:
        top = margin + max(offset_r, -1)
        bottom = image_size - margin - 1 + min(offset_r, 1)
        left = margin + max(offset_c, -1)
        right = image_size - margin - 1 + min(offset_c, 1)
        mid_r = (top + bottom) // 2
        mid_c = (left + right) // 2
        draw_line(canvas, top, mid_c, mid_r, right)
        draw_line(canvas, mid_r, right, bottom, mid_c)
        draw_line(canvas, bottom, mid_c, mid_r, left)
        draw_line(canvas, mid_r, left, top, mid_c)
    elif prompt_id == 6:
        row = np.clip(center + offset_r, margin, image_size - margin - 1)
        col = np.clip(center + offset_c, margin, image_size - margin - 1)
        draw_line(canvas, margin, col, image_size - margin - 1, col)
        draw_line(canvas, row, margin + 2, row, image_size - margin - 2)
    elif prompt_id == 7:
        top = np.clip(margin + offset_r, 1, image_size - 6)
        left = np.clip(margin + offset_c, 1, image_size - 6)
        bottom = min(image_size - 2, top + int(rng.integers(8, 11)))
        right = min(image_size - 2, left + int(rng.integers(8, 11)))
        draw_line(canvas, top, left, top, right)
        draw_line(canvas, bottom, left, bottom, right)
        draw_line(canvas, top, left, bottom, left)
        draw_line(canvas, top, right, bottom, right)
        inner_margin = 2
        canvas[top + inner_margin:bottom - inner_margin + 1, left + inner_margin:right - inner_margin + 1] = 0.0

    canvas = thicken(canvas, thickness - 1)
    canvas += rng.normal(0.0, 0.04, size=canvas.shape).astype(np.float32)
    return clamp01(canvas)


def make_dataset(samples_per_prompt, image_size, seed):
    rng = np.random.default_rng(seed)
    images = []
    prompt_ids = []
    for prompt_id in range(len(PROMPTS)):
        for _ in range(samples_per_prompt):
            images.append(render_prompt(prompt_id, image_size=image_size, rng=rng))
            prompt_ids.append(prompt_id)
    images = np.stack(images).astype(np.float32)
    prompt_ids = np.asarray(prompt_ids, dtype=np.int64)
    return images, prompt_ids


class TinyAutoencoder(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=128, latent_dim=32):
        super().__init__()
        self.enc1 = nn.Linear(input_dim, hidden_dim)
        self.enc2 = nn.Linear(hidden_dim, latent_dim)
        self.dec1 = nn.Linear(latent_dim, hidden_dim)
        self.dec2 = nn.Linear(hidden_dim, input_dim)
        self.relu = nn.ReLU()

    def encode(self, x):
        return self.enc2(self.relu(self.enc1(x)))

    def decode(self, z):
        return self.dec2(self.relu(self.dec1(z)))

    def forward(self, x):
        return self.decode(self.encode(x))


class TinyDenoiser(nn.Module):
    def __init__(self, latent_dim, prompt_dim, time_dim, hidden_dim=128):
        super().__init__()
        input_dim = latent_dim + prompt_dim + time_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, latent_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)


def batch_indices(num_items, batch_size, rng):
    indices = np.arange(num_items)
    rng.shuffle(indices)
    for start in range(0, num_items, batch_size):
        yield indices[start:start + batch_size]


def train_autoencoder(model, images, steps, batch_size, learning_rate, seed):
    optimizer = Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    rng = np.random.default_rng(seed)
    flat_images = images.reshape(images.shape[0], -1).astype(np.float32)
    losses = []

    for step in range(steps):
        batch = flat_images[rng.integers(0, flat_images.shape[0], size=batch_size)]
        x = nt.Tensor(batch)
        recon = model(x)
        loss = criterion(recon, x)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.data))

    return losses


def encode_dataset(model, images):
    flat_images = images.reshape(images.shape[0], -1).astype(np.float32)
    latents = model.encode(nt.Tensor(flat_images))
    return latents.data.astype(np.float32)


def build_denoiser_inputs(noisy_latents, prompt_ids, timesteps, num_steps, time_dim):
    prompt_features = prompt_one_hot(prompt_ids, len(PROMPTS))
    time_features = timestep_embedding(timesteps / max(num_steps - 1, 1), time_dim)
    features = np.concatenate([noisy_latents, prompt_features, time_features], axis=1)
    return features.astype(np.float32)


def train_diffusion(model, latents, prompt_ids, num_steps, train_steps, batch_size, learning_rate, seed, time_dim):
    optimizer = Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    rng = np.random.default_rng(seed)

    betas = np.linspace(1e-4, 0.02, num_steps, dtype=np.float32)
    alphas = 1.0 - betas
    alpha_bars = np.cumprod(alphas)
    losses = []

    for _ in range(train_steps):
        batch_idx = rng.integers(0, latents.shape[0], size=batch_size)
        z0 = latents[batch_idx]
        prompt_batch = prompt_ids[batch_idx]
        timesteps = rng.integers(0, num_steps, size=batch_size)
        noise = rng.normal(0.0, 1.0, size=z0.shape).astype(np.float32)

        alpha_bar_t = alpha_bars[timesteps].reshape(-1, 1)
        noisy_latents = np.sqrt(alpha_bar_t) * z0 + np.sqrt(1.0 - alpha_bar_t) * noise
        features = build_denoiser_inputs(noisy_latents, prompt_batch, timesteps, num_steps, time_dim)

        prediction = model(nt.Tensor(features))
        target = nt.Tensor(noise)
        loss = criterion(prediction, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.data))

    return losses, betas, alphas, alpha_bars


def sample_latents(model, autoencoder, betas, alphas, alpha_bars, steps, samples_per_prompt, latent_dim, time_dim, image_size, seed):
    rng = np.random.default_rng(seed)
    generated = []

    for prompt_id in range(len(PROMPTS)):
        z = rng.normal(0.0, 1.0, size=(samples_per_prompt, latent_dim)).astype(np.float32)
        prompt_batch = np.full(samples_per_prompt, prompt_id, dtype=np.int64)

        for timestep in reversed(range(steps)):
            t = np.full(samples_per_prompt, timestep, dtype=np.int64)
            features = build_denoiser_inputs(z, prompt_batch, t, steps, time_dim)
            pred_noise = model(nt.Tensor(features)).data.astype(np.float32)

            alpha_t = alphas[timestep]
            alpha_bar_t = alpha_bars[timestep]
            beta_t = betas[timestep]
            mean = (z - ((1.0 - alpha_t) / math.sqrt(max(1.0 - alpha_bar_t, 1e-6))) * pred_noise) / math.sqrt(alpha_t)

            if timestep > 0:
                z = mean + math.sqrt(beta_t) * rng.normal(0.0, 1.0, size=z.shape).astype(np.float32)
            else:
                z = mean

        decoded = autoencoder.decode(nt.Tensor(z)).data.astype(np.float32)
        decoded = decoded.reshape(samples_per_prompt, image_size, image_size)
        generated.append(clamp01(decoded))

    return generated


def save_grid(generated, output_path):
    rows = len(generated)
    cols = generated[0].shape[0]
    fig, axes = plt.subplots(rows, cols, figsize=(2.1 * cols, 2.1 * rows))
    if rows == 1:
        axes = np.asarray([axes])
    for row, images in enumerate(generated):
        for col in range(cols):
            ax = axes[row, col]
            ax.imshow(images[col], cmap="gray", vmin=0.0, vmax=1.0)
            if col == 0:
                ax.set_ylabel(PROMPTS[row], rotation=0, ha="right", va="center", labelpad=40)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle("Nano Stable Diffusion Samples", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Tiny Stable Diffusion-inspired project for NanoTorch.")
    parser.add_argument("--image-size", type=int, default=16)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--time-dim", type=int, default=16)
    parser.add_argument("--samples-per-prompt", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--autoencoder-steps", type=int, default=300)
    parser.add_argument("--diffusion-train-steps", type=int, default=600)
    parser.add_argument("--diffusion-steps", type=int, default=20)
    parser.add_argument("--autoencoder-lr", type=float, default=3e-3)
    parser.add_argument("--diffusion-lr", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    seed_everything(args.seed)
    project_dir = Path(__file__).resolve().parent
    artifact_dir = project_dir / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    images, prompt_ids = make_dataset(
        samples_per_prompt=args.samples_per_prompt,
        image_size=args.image_size,
        seed=args.seed,
    )

    autoencoder = TinyAutoencoder(
        input_dim=args.image_size * args.image_size,
        hidden_dim=128,
        latent_dim=args.latent_dim,
    )
    denoiser = TinyDenoiser(
        latent_dim=args.latent_dim,
        prompt_dim=len(PROMPTS),
        time_dim=args.time_dim,
        hidden_dim=128,
    )

    autoencoder_losses = train_autoencoder(
        autoencoder,
        images,
        steps=args.autoencoder_steps,
        batch_size=args.batch_size,
        learning_rate=args.autoencoder_lr,
        seed=args.seed,
    )

    latents = encode_dataset(autoencoder, images)
    diffusion_losses, betas, alphas, alpha_bars = train_diffusion(
        denoiser,
        latents,
        prompt_ids,
        num_steps=args.diffusion_steps,
        train_steps=args.diffusion_train_steps,
        batch_size=args.batch_size,
        learning_rate=args.diffusion_lr,
        seed=args.seed + 1,
        time_dim=args.time_dim,
    )

    generated = sample_latents(
        denoiser,
        autoencoder,
        betas,
        alphas,
        alpha_bars,
        steps=args.diffusion_steps,
        samples_per_prompt=2,
        latent_dim=args.latent_dim,
        time_dim=args.time_dim,
        image_size=args.image_size,
        seed=args.seed + 2,
    )
    sample_path = artifact_dir / "generated_samples.png"
    save_grid(generated, sample_path)

    history = {
        "project": "nano_stable_diffusion",
        "prompts": PROMPTS,
        "config": {
            "image_size": args.image_size,
            "latent_dim": args.latent_dim,
            "time_dim": args.time_dim,
            "samples_per_prompt": args.samples_per_prompt,
            "batch_size": args.batch_size,
            "autoencoder_steps": args.autoencoder_steps,
            "diffusion_train_steps": args.diffusion_train_steps,
            "diffusion_steps": args.diffusion_steps,
            "autoencoder_lr": args.autoencoder_lr,
            "diffusion_lr": args.diffusion_lr,
            "seed": args.seed,
        },
        "metrics": {
            "autoencoder_initial_loss": autoencoder_losses[0],
            "autoencoder_final_loss": autoencoder_losses[-1],
            "diffusion_initial_loss": diffusion_losses[0],
            "diffusion_final_loss": diffusion_losses[-1],
        },
    }

    history_path = artifact_dir / "training_history.json"
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

    print("Nano Stable Diffusion project complete")
    print(f"Prompts: {len(PROMPTS)}")
    print(f"Autoencoder final loss: {autoencoder_losses[-1]:.6f}")
    print(f"Diffusion final loss: {diffusion_losses[-1]:.6f}")
    print(f"Samples saved to: {sample_path}")
    print(f"History saved to: {history_path}")


if __name__ == "__main__":
    main()
