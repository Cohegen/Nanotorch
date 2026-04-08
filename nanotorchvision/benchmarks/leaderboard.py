import csv
import json
from pathlib import Path


def discover_summary_files(root_dir):
    """Find benchmark summary JSON files under a directory."""
    root = Path(root_dir)
    return sorted(root.rglob("*_summary.json"))


def load_summary_file(path):
    """Load a single benchmark summary and normalize key leaderboard fields."""
    summary_path = Path(path)
    with summary_path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)

    model_name = summary.get("model_name")
    if model_name is None:
        model_name = summary_path.stem.replace("_summary", "")

    runtime_seconds = summary.get("total_training_time_seconds")
    if runtime_seconds is None:
        runtime_seconds = summary.get("training_time_seconds")

    record = {
        "model_name": model_name,
        "dataset": summary.get("dataset", "unknown"),
        "epochs": summary.get("epochs"),
        "batch_size": summary.get("batch_size"),
        "parameter_count": summary.get("parameter_count"),
        "final_test_acc": summary.get("final_test_acc"),
        "best_test_acc": summary.get("best_test_acc", summary.get("final_test_acc")),
        "final_test_loss": summary.get("final_test_loss"),
        "train_size": summary.get("train_size"),
        "test_size": summary.get("test_size"),
        "runtime_seconds": runtime_seconds,
        "summary_path": str(summary_path),
    }
    return record


def build_leaderboard(summary_paths):
    """Build a sorted leaderboard from summary JSON paths."""
    rows = [load_summary_file(path) for path in summary_paths]

    def sort_key(row):
        best_test_acc = row["best_test_acc"]
        final_test_acc = row["final_test_acc"]
        runtime_seconds = row["runtime_seconds"]

        if best_test_acc is None:
            best_test_acc = -1.0
        if final_test_acc is None:
            final_test_acc = -1.0
        if runtime_seconds is None:
            runtime_seconds = float("inf")

        return (-best_test_acc, -final_test_acc, runtime_seconds, row["model_name"])

    rows.sort(key=sort_key)
    for index, row in enumerate(rows, start=1):
        row["rank"] = index
    return rows


def write_leaderboard_csv(rows, output_path):
    """Write leaderboard rows to CSV."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "rank",
        "model_name",
        "dataset",
        "epochs",
        "batch_size",
        "parameter_count",
        "best_test_acc",
        "final_test_acc",
        "final_test_loss",
        "runtime_seconds",
        "train_size",
        "test_size",
        "summary_path",
    ]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_leaderboard_markdown(rows, output_path):
    """Write leaderboard rows to a Markdown table."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# NanoTorchVision Leaderboard",
        "",
        "| Rank | Model | Dataset | Params | Best Test Acc | Final Test Acc | Final Test Loss | Runtime (s) | Epochs | Batch Size |",
        "| :---: | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for row in rows:
        best_test_acc = _format_metric(row["best_test_acc"], percentage=True)
        final_test_acc = _format_metric(row["final_test_acc"], percentage=True)
        final_test_loss = _format_metric(row["final_test_loss"])
        runtime_seconds = _format_metric(row["runtime_seconds"])
        epochs = _format_metric(row["epochs"], integer=True)
        batch_size = _format_metric(row["batch_size"], integer=True)
        parameter_count = _format_metric(row["parameter_count"], integer=True)

        lines.append(
            f"| {row['rank']} | {row['model_name']} | {row['dataset']} | "
            f"{parameter_count} | {best_test_acc} | {final_test_acc} | {final_test_loss} | "
            f"{runtime_seconds} | {epochs} | {batch_size} |"
        )

    with output.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def _format_metric(value, percentage=False, integer=False):
    if value is None:
        return "-"
    if integer:
        return str(int(value))
    if percentage:
        return f"{100.0 * float(value):.1f}%"
    return f"{float(value):.4f}"
