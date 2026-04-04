import os
import sys
import numpy as np

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import nanotorch as nt
import nanotorch.nn as nn
from nanotorch import Tensor

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

console = Console()


class Perceptron:
    """
    A simple implementation of the perceptron = Linear + Sigmoid
    """

    def __init__(self, input_size=2, output_size=1):
        self.linear = nn.Linear(input_size, output_size)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x = self.linear(x)
        x = self.activation(x)
        return x

    def __call__(self, x):
        return self.forward(x)


