from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nanotorchvision.benchmarks import (
    build_leaderboard,
    discover_summary_files,
    write_leaderboard_csv,
    write_leaderboard_markdown,
)


def main():
    plot_dir = Path(__file__).resolve().parent / "plots"
    summary_files = discover_summary_files(plot_dir)

    if not summary_files:
        print(f"No benchmark summaries found in {plot_dir}")
        return []

    rows = build_leaderboard(summary_files)
    csv_path = plot_dir / "leaderboard.csv"
    md_path = plot_dir / "leaderboard.md"

    write_leaderboard_csv(rows, csv_path)
    write_leaderboard_markdown(rows, md_path)

    print(f"Wrote leaderboard CSV to {csv_path}")
    print(f"Wrote leaderboard Markdown to {md_path}")
    for row in rows:
        best = row["best_test_acc"]
        best_text = "-" if best is None else f"{100.0 * float(best):.1f}%"
        print(f"#{row['rank']} {row['model_name']} ({row['dataset']}): best_test_acc={best_text}")

    return rows


if __name__ == "__main__":
    main()
