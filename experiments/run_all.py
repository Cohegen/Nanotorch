from pathlib import Path
import subprocess
import sys


def main():
    experiments_dir = Path(__file__).resolve().parent
    scripts = sorted(
        path for path in experiments_dir.glob("test_*.py")
        if path.name != "run_all.py"
    )

    if not scripts:
        print("No experiment scripts found.")
        return 1

    failures = []
    for script in scripts:
        print(f"Running {script.name}...")
        result = subprocess.run([sys.executable, str(script)], check=False)
        if result.returncode != 0:
            failures.append(script.name)

    if failures:
        print("\nFailed experiments:")
        for name in failures:
            print(f"- {name}")
        return 1

    print("\nAll NanoTorch experiments passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
