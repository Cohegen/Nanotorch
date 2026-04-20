"""Unit tests for the training module."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
from Tensor.tensor import Tensor
from layers.layers import Linear
from optimizers.optimizers import SGD
from training.training import CosineSchedule, clip_grad_norm
from losses.losses import MSELoss
from training.training import Trainer


class TestCosineSchedule:
    def test_initial_lr(self):
        schedule = CosineSchedule(max_lr=0.01, min_lr=0.001, total_epochs=100)
        lr = schedule.get_lr(epoch=0)
        # At epoch 0, lr should be max_lr (cosine starts at peak)
        np.testing.assert_almost_equal(lr, 0.01, decimal=4)

    def test_cosine_decay(self):
        schedule = CosineSchedule(max_lr=0.01, min_lr=0.001, total_epochs=100)
        lr0 = schedule.get_lr(epoch=0)
        lr50 = schedule.get_lr(epoch=50)
        lr99 = schedule.get_lr(epoch=99)
        # LR should decrease over epochs
        assert lr0 >= lr50
        assert lr50 >= lr99

    def test_min_lr_bound(self):
        schedule = CosineSchedule(max_lr=0.01, min_lr=0.001, total_epochs=100)
        lr = schedule.get_lr(epoch=99)
        assert lr >= 0.001 - 1e-6

    def test_past_total_epochs(self):
        schedule = CosineSchedule(max_lr=0.01, min_lr=0.001, total_epochs=100)
        lr = schedule.get_lr(epoch=200)
        np.testing.assert_almost_equal(lr, 0.001, decimal=4)


class TestClipGradNorm:
    def test_clips_large_gradient(self):
        param = Tensor([1.0], requires_grad=True)
        param.grad = Tensor([100.0])
        clip_grad_norm([param], max_norm=1.0)
        norm = np.linalg.norm(param.grad.data)
        np.testing.assert_almost_equal(norm, 1.0, decimal=5)

    def test_no_clip_small_gradient(self):
        param = Tensor([1.0], requires_grad=True)
        param.grad = Tensor([0.5])
        clip_grad_norm([param], max_norm=1.0)
        np.testing.assert_almost_equal(param.grad.data[0], 0.5, decimal=5)

    def test_multiple_params(self):
        p1 = Tensor([1.0], requires_grad=True)
        p1.grad = Tensor([30.0])
        p2 = Tensor([1.0], requires_grad=True)
        p2.grad = Tensor([40.0])
        total_norm = clip_grad_norm([p1, p2], max_norm=5.0)
        # clip_grad_norm returns the computed norm and clips gradients
        assert total_norm > 0
        # After clipping, gradients should be scaled down
        assert abs(p1.grad.data[0]) <= 30.0

    def test_skips_none_grad(self):
        p1 = Tensor([1.0], requires_grad=True)
        p1.grad = Tensor([10.0])
        p2 = Tensor([1.0], requires_grad=True)
        p2.grad = None
        # Should not raise
        clip_grad_norm([p1, p2], max_norm=1.0)


class _SingleLinearModel:
    def __init__(self):
        self.layer = Linear(1, 1)

    def forward(self, x):
        return self.layer(x)

    def parameters(self):
        return self.layer.parameters()


class TestTrainerValidation:
    def test_train_epoch_updates_history(self):
        model = _SingleLinearModel()
        optimizer = SGD(model.parameters(), lr=0.01)
        trainer = Trainer(model, optimizer, MSELoss(), grad_clip_norm=1.0)

        x = Tensor(np.array([[1.0], [2.0]], dtype=np.float32))
        y = Tensor(np.array([[0.0], [1.0]], dtype=np.float32))
        dataloader = [(x, y)]

        avg_loss = trainer.train_epoch(dataloader)

        assert avg_loss >= 0.0
        assert len(trainer.history["train_loss"]) == 1

    def test_evaluate_raises_on_nonfinite_loss(self):
        class BadLoss:
            def forward(self, outputs, targets):
                return Tensor(np.array(np.nan, dtype=np.float32))

        model = _SingleLinearModel()
        optimizer = SGD(model.parameters(), lr=0.01)
        trainer = Trainer(model, optimizer, BadLoss(), raise_on_nonfinite=True)

        x = Tensor(np.array([[1.0]], dtype=np.float32))
        y = Tensor(np.array([[0.0]], dtype=np.float32))

        with pytest.raises(ValueError):
            trainer.evaluate([(x, y)])
