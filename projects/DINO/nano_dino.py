import argparse
import json
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
from autograd.autograd import Function
from Tensor import Tensor
from nanotorchvision.datasets import load_nanodigits


class DinoCrossEntropyBackward(Function):
    def __init__(self, student_logits, teacher_probs, student_temp):
        super().__init__(student_logits)
        self.teacher_probs = teacher_probs.astype(np.float32)
        self.student_temp = float(student_temp)
        scaled = student_logits.data / self.student_temp
        shifted = scaled - np.max(scaled, axis=1, keepdims=True)
        exp_scaled = np.exp(shifted)
        self.student_probs = exp_scaled / np.sum(exp_scaled, axis=1, keepdims=True)

    def apply(self, grad_output):
        student_logits, = self.saved_tensors
        grad_logits = None
        if getattr(student_logits, "requires_grad", False):
            batch_size = max(student_logits.shape[0], 1)
            scale = float(np.asarray(grad_output).reshape(-1)[0])
            grad_logits = scale * (self.student_probs - self.teacher_probs) / (batch_size * self.student_temp)
        return (grad_logits,)


def dino_cross_entropy(student_logits, teacher_probs, student_temp):
    scaled = student_logits.data / student_temp
    shifted = scaled - np.max(scaled, axis=1, keepdims=True)
    log_probs = shifted - np.log(np.sum(np.exp(shifted), axis=1, keepdims=True) + 1e-9)
    loss_value = -np.mean(np.sum(teacher_probs * log_probs, axis=1))
    loss = Tensor(np.array(loss_value, dtype=np.float32))
    if getattr(student_logits, "requires_grad", False):
        loss.requires_grad = True
        loss._grad_fn = DinoCrossEntropyBackward(student_logits, teacher_probs, student_temp)
    return loss


def count_parameters(model):
    return int(sum(np.prod(param.data.shape) for param in model.parameters()))


def seed_everything(seed):
    np.random.seed(seed)


def flatten_images(images):
    return images.reshape(images.shape[0], -1).astype(np.float32)


def shift_image(image, shift_y, shift_x):
    shifted = np.roll(image, shift_y, axis=0)
    shifted = np.roll(shifted, shift_x, axis=1)
    if shift_y > 0:
        shifted[:shift_y, :] = 0.0
    elif shift_y < 0:
        shifted[shift_y:, :] = 0.0
    if shift_x > 0:
        shifted[:, :shift_x] = 0.0
    elif shift_x < 0:
        shifted[:, shift_x:] = 0.0
    return shifted


def augment_batch(images, rng):
    augmented = images.copy().astype(np.float32)
    batch_size = augmented.shape[0]

    for index in range(batch_size):
        image = augmented[index]

        if rng.random() < 0.9:
            shift_y = int(rng.integers(-1, 2))
            shift_x = int(rng.integers(-1, 2))
            image = shift_image(image, shift_y, shift_x)

        if rng.random() < 0.5:
            image = np.fliplr(image)

        if rng.random() < 0.35:
            cutout_h = int(rng.integers(1, 3))
            cutout_w = int(rng.integers(1, 3))
            top = int(rng.integers(0, max(1, image.shape[0] - cutout_h + 1)))
            left = int(rng.integers(0, max(1, image.shape[1] - cutout_w + 1)))
            image[top:top + cutout_h, left:left + cutout_w] = 0.0

        noise = rng.normal(0.0, 0.08, size=image.shape).astype(np.float32)
        gain = float(rng.uniform(0.85, 1.15))
        bias = float(rng.uniform(-0.1, 0.1))
        image = np.clip(image * gain + bias + noise, 0.0, 1.0)
        augmented[index] = image

    return augmented


def two_views(images, rng):
    return augment_batch(images, rng), augment_batch(images, rng)


class TinyBackbone(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=128, feature_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, feature_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)


class ProjectionHead(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=128, output_dim=32):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)


class DinoNetwork(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=128, feature_dim=64, projection_dim=32):
        super().__init__()
        self.backbone = TinyBackbone(input_dim=input_dim, hidden_dim=hidden_dim, feature_dim=feature_dim)
        self.head = ProjectionHead(input_dim=feature_dim, hidden_dim=hidden_dim, output_dim=projection_dim)

    def embed(self, x):
        return self.backbone(x)

    def forward(self, x):
        features = self.embed(x)
        return self.head(features)


def copy_student_to_teacher(student, teacher):
    teacher.load_state_dict(student.state_dict())
    for param in teacher.parameters():
        param.requires_grad = False
        param.grad = None


def ema_update(student, teacher, momentum):
    for student_param, teacher_param in zip(student.parameters(), teacher.parameters()):
        teacher_param.data = momentum * teacher_param.data + (1.0 - momentum) * student_param.data


def teacher_distribution(logits, center, teacher_temp):
    centered = (logits.data - center) / teacher_temp
    centered = centered - np.max(centered, axis=1, keepdims=True)
    exp_centered = np.exp(centered)
    return exp_centered / np.sum(exp_centered, axis=1, keepdims=True)


def update_center(center, teacher_probs, momentum):
    batch_center = np.mean(teacher_probs, axis=0, keepdims=True)
    return center * momentum + batch_center * (1.0 - momentum)


def l2_normalize(array):
    norms = np.linalg.norm(array, axis=1, keepdims=True) + 1e-9
    return array / norms


def extract_embeddings(model, images, batch_size):
    flat = flatten_images(images)
    embeddings = []
    for start in range(0, flat.shape[0], batch_size):
        batch = flat[start:start + batch_size]
        features = model.embed(nt.Tensor(batch)).data.astype(np.float32)
        embeddings.append(features)
    return l2_normalize(np.concatenate(embeddings, axis=0))


def centroid_probe(train_embeddings, train_labels, test_embeddings, test_labels):
    classes = np.unique(train_labels)
    centroids = []
    for class_id in classes:
        centroids.append(np.mean(train_embeddings[train_labels == class_id], axis=0))
    centroids = l2_normalize(np.stack(centroids).astype(np.float32))

    scores = test_embeddings @ centroids.T
    predictions = classes[np.argmax(scores, axis=1)]
    accuracy = float(np.mean(predictions == test_labels))
    return accuracy


def plot_history(history, output_path):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    axes[0].plot(history["epoch"], history["loss"], marker="o")
    axes[0].set_title("DINO Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")

    axes[1].plot(history["epoch"], history["teacher_entropy"], marker="o")
    axes[1].set_title("Teacher Entropy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Entropy")

    axes[2].plot(history["epoch"], history["probe_accuracy"], marker="o")
    axes[2].set_title("Centroid Probe Accuracy")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Accuracy")

    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def train_epoch(student, teacher, optimizer, images, batch_size, rng, center, student_temp, teacher_temp, teacher_momentum, center_momentum):
    indices = np.arange(images.shape[0])
    rng.shuffle(indices)
    batch_losses = []
    batch_entropies = []

    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start:start + batch_size]
        batch_images = images[batch_indices]
        view_a, view_b = two_views(batch_images, rng)

        student_a = student(nt.Tensor(flatten_images(view_a)))
        student_b = student(nt.Tensor(flatten_images(view_b)))

        teacher_a_logits = teacher(nt.Tensor(flatten_images(view_a)))
        teacher_b_logits = teacher(nt.Tensor(flatten_images(view_b)))

        teacher_a_probs = teacher_distribution(teacher_a_logits, center, teacher_temp)
        teacher_b_probs = teacher_distribution(teacher_b_logits, center, teacher_temp)

        loss_ab = dino_cross_entropy(student_a, teacher_b_probs, student_temp)
        loss_ba = dino_cross_entropy(student_b, teacher_a_probs, student_temp)
        loss = (loss_ab + loss_ba) * 0.5

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ema_update(student, teacher, teacher_momentum)

        combined_teacher = np.concatenate([teacher_a_probs, teacher_b_probs], axis=0)
        center = update_center(center, combined_teacher, center_momentum)

        entropy = -np.mean(np.sum(combined_teacher * np.log(combined_teacher + 1e-9), axis=1))
        batch_losses.append(float(loss.data))
        batch_entropies.append(float(entropy))

    return float(np.mean(batch_losses)), float(np.mean(batch_entropies)), center


def main():
    parser = argparse.ArgumentParser(description="NanoTorch-native DINO-style self-supervised training on NanoDigits.")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--feature-dim", type=int, default=64)
    parser.add_argument("--projection-dim", type=int, default=32)
    parser.add_argument("--student-temp", type=float, default=0.1)
    parser.add_argument("--teacher-temp", type=float, default=0.04)
    parser.add_argument("--teacher-momentum", type=float, default=0.996)
    parser.add_argument("--center-momentum", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    seed_everything(args.seed)
    artifact_dir = Path(__file__).resolve().parent / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    (train_images, train_labels), (test_images, test_labels) = load_nanodigits()
    student = DinoNetwork(
        input_dim=64,
        hidden_dim=args.hidden_dim,
        feature_dim=args.feature_dim,
        projection_dim=args.projection_dim,
    )
    teacher = DinoNetwork(
        input_dim=64,
        hidden_dim=args.hidden_dim,
        feature_dim=args.feature_dim,
        projection_dim=args.projection_dim,
    )
    copy_student_to_teacher(student, teacher)

    optimizer = Adam(student.parameters(), lr=args.learning_rate)
    rng = np.random.default_rng(args.seed)
    center = np.zeros((1, args.projection_dim), dtype=np.float32)

    history = {
        "epoch": [],
        "loss": [],
        "teacher_entropy": [],
        "probe_accuracy": [],
    }

    print("Training NanoDINO on NanoDigits")
    print(f"Train set: {train_images.shape}, Test set: {test_images.shape}")
    print(
        f"Hyperparameters -> epochs={args.epochs}, batch_size={args.batch_size}, "
        f"lr={args.learning_rate}, teacher_momentum={args.teacher_momentum}"
    )
    print(f"Student parameters: {count_parameters(student)}")

    for epoch in range(args.epochs):
        loss_value, teacher_entropy, center = train_epoch(
            student,
            teacher,
            optimizer,
            train_images,
            batch_size=args.batch_size,
            rng=rng,
            center=center,
            student_temp=args.student_temp,
            teacher_temp=args.teacher_temp,
            teacher_momentum=args.teacher_momentum,
            center_momentum=args.center_momentum,
        )

        train_embeddings = extract_embeddings(student, train_images, args.batch_size)
        test_embeddings = extract_embeddings(student, test_images, args.batch_size)
        probe_accuracy = centroid_probe(train_embeddings, train_labels, test_embeddings, test_labels)

        history["epoch"].append(epoch + 1)
        history["loss"].append(loss_value)
        history["teacher_entropy"].append(teacher_entropy)
        history["probe_accuracy"].append(probe_accuracy)

        print(
            f"Epoch {epoch + 1:02d}/{args.epochs} | "
            f"loss={loss_value:.4f} | teacher_entropy={teacher_entropy:.4f} | "
            f"probe_acc={probe_accuracy:.3f}"
        )

    curves_path = artifact_dir / "training_curves.png"
    plot_history(history, curves_path)

    summary = {
        "project": "nano_dino",
        "dataset": "NanoDigits",
        "config": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "hidden_dim": args.hidden_dim,
            "feature_dim": args.feature_dim,
            "projection_dim": args.projection_dim,
            "student_temp": args.student_temp,
            "teacher_temp": args.teacher_temp,
            "teacher_momentum": args.teacher_momentum,
            "center_momentum": args.center_momentum,
            "seed": args.seed,
        },
        "metrics": {
            "final_loss": history["loss"][-1],
            "final_teacher_entropy": history["teacher_entropy"][-1],
            "final_probe_accuracy": history["probe_accuracy"][-1],
            "best_probe_accuracy": max(history["probe_accuracy"]),
        },
        "student_parameter_count": count_parameters(student),
    }

    history_path = artifact_dir / "training_history.json"
    history_path.write_text(json.dumps({"history": history, "summary": summary}, indent=2), encoding="utf-8")

    print("NanoDINO project complete")
    print(f"Final probe accuracy: {history['probe_accuracy'][-1]:.3f}")
    print(f"Artifacts saved to: {artifact_dir}")


if __name__ == "__main__":
    main()
