import nanotorchvision as ntv


def test_nanotorchvision_package_layout():
    assert ntv.datasets.load_nanodigits is not None
    assert ntv.benchmarks.build_leaderboard is not None

    assert ntv.models.MiniResNetDigits is not None
    assert ntv.models.AlexNetTinyDigits is not None
    assert ntv.models.MobileNetStyleTinyDigits is not None
    assert ntv.models.VGGNetTinyDigits is not None
    assert ntv.models.ViTTinyDigits is not None
    assert ntv.models.MODEL_REGISTRY["mini_resnet_digits"] is ntv.models.MiniResNetDigits
    assert ntv.models.MODEL_REGISTRY["vggnet_tiny_digits"] is ntv.models.VGGNetTinyDigits
