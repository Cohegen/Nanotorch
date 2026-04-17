import copy
import sys
from pathlib import Path

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from optimization.profiling.profiling import Profiler
from optimization.quantization.quantization import QuantizedLinear, quantize_model
from common import (
    build_digits_mlp,
    load_flat_nanodigits,
    make_calibration_samples,
    make_flat_loader,
    next_batch,
    save_json,
    seed_everything,
    train_classifier,
    evaluate_classifier,
)


def estimate_fp32_bytes(model):
    return int(sum(param.data.size for param in model.parameters()) * 4)


def estimate_quantized_bytes(model):
    total = 0
    for layer in getattr(model, "layers", []):
        if isinstance(layer, QuantizedLinear):
            total += int(layer.memory_usage()["quantized_bytes"])
        elif hasattr(layer, "parameters"):
            total += int(sum(param.data.size for param in layer.parameters()) * 4)
    return total


def run_project(epochs=12, batch_size=32, learning_rate=0.05, momentum=0.9, seed=7):
    seed_everything(seed)
    (train_x, train_y), (test_x, test_y) = load_flat_nanodigits()

    model = build_digits_mlp(hidden_dims=(128, 64))
    history = train_classifier(
        model,
        train_x,
        train_y,
        test_x,
        test_y,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        momentum=momentum,
        seed=seed,
    )

    test_loader = make_flat_loader(test_x, test_y, batch_size=batch_size, shuffle=False, seed=None)
    baseline_metrics = evaluate_classifier(model, test_loader)
    sample_inputs, _ = next_batch(test_loader)

    profiler = Profiler()
    baseline_latency_ms = profiler.measure_latency(model, sample_inputs, warmup=3, iterations=10)
    baseline_bytes = estimate_fp32_bytes(model)

    quantized_model = copy.deepcopy(model)
    calibration_samples = make_calibration_samples(train_x, max_samples=32)
    quantize_model(quantized_model, calibration_samples)

    quantized_loader = make_flat_loader(test_x, test_y, batch_size=batch_size, shuffle=False, seed=None)
    quantized_metrics = evaluate_classifier(quantized_model, quantized_loader)
    quantized_sample_inputs, _ = next_batch(quantized_loader)
    quantized_latency_ms = profiler.measure_latency(quantized_model, quantized_sample_inputs, warmup=3, iterations=10)
    quantized_bytes = estimate_quantized_bytes(quantized_model)

    results = {
        "project": "quantized_nanodigits",
        "dataset": "NanoDigits",
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "momentum": momentum,
        "final_training_epoch": history[-1],
        "baseline": {
            "test_loss": baseline_metrics["loss"],
            "test_accuracy": baseline_metrics["accuracy"],
            "latency_ms": baseline_latency_ms,
            "estimated_model_bytes": baseline_bytes,
        },
        "quantized": {
            "test_loss": quantized_metrics["loss"],
            "test_accuracy": quantized_metrics["accuracy"],
            "latency_ms": quantized_latency_ms,
            "estimated_model_bytes": quantized_bytes,
        },
        "comparison": {
            "accuracy_delta": quantized_metrics["accuracy"] - baseline_metrics["accuracy"],
            "latency_speedup_x": baseline_latency_ms / max(quantized_latency_ms, 1e-9),
            "size_reduction_ratio": baseline_bytes / max(quantized_bytes, 1),
        },
    }
    return results


def main():
    project_dir = Path(__file__).resolve().parent
    output_path = project_dir / "reports" / "quantized_nanodigits.json"
    results = run_project()
    save_json(output_path, results)

    print("Quantized NanoDigits project complete")
    print(f"Baseline accuracy: {results['baseline']['test_accuracy']:.3f}")
    print(f"Quantized accuracy: {results['quantized']['test_accuracy']:.3f}")
    print(f"Latency speedup: {results['comparison']['latency_speedup_x']:.2f}x")
    print(f"Estimated size reduction: {results['comparison']['size_reduction_ratio']:.2f}x")
    print(f"Report saved to: {output_path}")


if __name__ == "__main__":
    main()
