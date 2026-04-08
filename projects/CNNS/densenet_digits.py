import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark_utils import run_nanodigits_benchmark
from nanotorchvision.models import DenseNetTinyDigits


def main(epochs=10, batch_size=32, learning_rate=0.02, momentum=0.9):
    model = DenseNetTinyDigits(num_classes=10, growth_rate=4)
    return run_nanodigits_benchmark(
        model=model,
        model_name="densenet_tiny_digits",
        title_prefix="DenseNet Tiny Digits",
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        momentum=momentum,
    )


if __name__ == "__main__":
    main()
