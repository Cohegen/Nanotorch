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

from layers.layers import Linear
from optimization.compression.compression import compress_model, low_rank_approximate, measure_sparsity
from optimization.profiling.profiling import Profiler
from common import (
    build_digits_mlp,
    load_flat_nanodigits,
    make_flat_loader,
    next_batch,
    save_json,
    seed_everything,
    train_classifier,
    evaluate_classifier,
)


def analyze_low_rank_layers(model, rank_ratio=0.5):
    analysis = []
    for index, layer in enumerate(getattr(model, "layers", [])):
        if isinstance(layer, Linear):
            weight = layer.weight.data
            u, s, v = low_rank_approximate(weight, rank_ratio=rank_ratio)
            original_params = int(weight.size)
            approximated_params = int(u.size + s.size + v.size)
            analysis.append(
                {
                    "layer_index": index,
                    "shape": list(weight.shape),
                    "target_rank": int(s.shape[0]),
                    "original_weight_params": original_params,
                    "approximated_params": approximated_params,
                    "estimated_reduction_ratio": original_params / max(approximated_params, 1),
                }
            )
    return analysis


def run_project(
    epochs=12,
    batch_size=32,
    learning_rate=0.05,
    momentum=0.9,
    seed=7,
    compression_config=None,
):
    if compression_config is None:
        compression_config = {"magnitude_prune": 0.6, "structured_prune": 0.2, "low_rank": 0.5}

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
    baseline_eval = evaluate_classifier(model, test_loader)
    sample_inputs, _ = next_batch(test_loader)

    profiler = Profiler()
    baseline_profile = profiler.profile_forward_pass(model, sample_inputs)

    compressed_model = copy.deepcopy(model)
    compression_stats = compress_model(compressed_model, compression_config)
    compressed_loader = make_flat_loader(test_x, test_y, batch_size=batch_size, shuffle=False, seed=None)
    compressed_eval = evaluate_classifier(compressed_model, compressed_loader)
    compressed_sample_inputs, _ = next_batch(compressed_loader)
    compressed_profile = profiler.profile_forward_pass(compressed_model, compressed_sample_inputs)

    low_rank_analysis = analyze_low_rank_layers(model, rank_ratio=compression_config.get("low_rank", 0.5))

    results = {
        "project": "compression_profiling_pipeline",
        "dataset": "NanoDigits",
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "momentum": momentum,
        "compression_config": compression_config,
        "final_training_epoch": history[-1],
        "baseline": {
            "evaluation": baseline_eval,
            "profile": baseline_profile,
            "sparsity_percent": measure_sparsity(model),
        },
        "compressed": {
            "evaluation": compressed_eval,
            "profile": compressed_profile,
            "sparsity_percent": measure_sparsity(compressed_model),
            "compression_stats": compression_stats,
        },
        "low_rank_analysis": low_rank_analysis,
        "comparison": {
            "accuracy_delta": compressed_eval["accuracy"] - baseline_eval["accuracy"],
            "latency_delta_ms": compressed_profile["latency_ms"] - baseline_profile["latency_ms"],
            "parameter_delta": compressed_profile["parameters"] - baseline_profile["parameters"],
            "peak_memory_delta_mb": compressed_profile["peak_memory_mb"] - baseline_profile["peak_memory_mb"],
        },
    }
    return results


def main():
    project_dir = Path(__file__).resolve().parent
    output_path = project_dir / "reports" / "compression_profiling_pipeline.json"
    results = run_project()
    save_json(output_path, results)

    print("Compression + profiling pipeline complete")
    print(f"Baseline accuracy: {results['baseline']['evaluation']['accuracy']:.3f}")
    print(f"Compressed accuracy: {results['compressed']['evaluation']['accuracy']:.3f}")
    print(f"Baseline sparsity: {results['baseline']['sparsity_percent']:.2f}%")
    print(f"Compressed sparsity: {results['compressed']['sparsity_percent']:.2f}%")
    print(f"Report saved to: {output_path}")


if __name__ == "__main__":
    main()
