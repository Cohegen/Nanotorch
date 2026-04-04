import nanotorch as nt


def test_nanotorch_package_layout():
    assert nt.Tensor is not None
    assert nt.tensor([1, 2, 3]).shape == (3,)

    assert nt.nn.Module is nt.nn.Layer
    assert nt.nn.Linear is nt.nn.modules.Linear
    assert nt.nn.Dropout is nt.nn.modules.Dropout

    assert nt.optim.SGD is not None
    assert nt.optim.Adam is not None
    assert nt.optim.AdamW is not None

    assert nt.utils.data.DataLoader is nt.utils.data.Dataloader
    assert nt.utils.data.Dataset is not None
    assert nt.autograd.enable_autograd is not None
