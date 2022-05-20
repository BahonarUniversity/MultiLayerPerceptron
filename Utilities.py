from enum import Enum

import numpy as np

from ActivationFunctions import Function, IdentityFunction


class LearningMode(Enum):
    Incremental = 1
    Batch = 2


class HiddenLayersStructure:

    def __init__(self):
        self.layers_size = []
        self.layers_activation = []

    def add_layer(self, size: int, activation: Function = IdentityFunction()):
        self.layers_size.append(size + 1) # one added for bias neuron
        self.layers_activation.append(activation)

    def __iter__(self):
        self.n = 0
        return self

    def __next__(self):
        if self.n < len(self.layers_size):
            self.n += 1
            return [self.layers_size[self.n-1], self.layers_activation[self.n-1]]
        else:
            raise StopIteration

    def __getitem__(self, index: int):
        return [self.layers_size[index], self.layers_activation[index]]

    def __len__(self):
        return len(self.layers_size)


def make_gaussian(min_value, max_value, grid_size):
    x, y = np.meshgrid(np.linspace(min_value, max_value, grid_size), np.linspace(min_value, max_value, grid_size))
    dst = np.sqrt(x * x + y * y)

    # Initializing sigma and muu
    sigma = 1
    muu = 0.000

    # Calculating Gaussian array
    gauss = np.exp(-((dst - muu) ** 2 / (2.0 * sigma ** 2)))

    return gauss