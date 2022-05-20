import math
from abc import ABC, abstractmethod
import numpy as np


def round_input(net_input: float, threshold: float = 20):
    return -threshold if net_input < -threshold else (threshold if net_input > threshold else net_input)


class Function(ABC):
    @abstractmethod
    def run(self, net_input: float):
        pass

    @abstractmethod
    def run_differential(self, net_input: float):
        pass


class ConstantFunction(Function):

    def run(self, net_input: float):
        return 1

    def run_differential(self, net_input: float):
        return 1


class IdentityFunction(Function):

    def run(self, net_input: float):
        return net_input

    def run_differential(self, net_input: float):
        return 1


class BinaryFunction(Function):
    def run(self, net_input: float):
        return 0 if net_input < 0 else 1

    def run_differential(self, net_input: float):
        return 1


class BipolarFunction(Function):

    def __init__(self, factor: float = 1):
        self.factor = factor

    def run(self, net_input: float):
        return self.factor * (-1 if net_input < 0 else 1)

    def run_differential(self, net_input: float):
        return self.factor


class ReluFunction(Function):
    def run(self, net_input: float):
        return 0 if net_input < 0 else net_input

    def run_differential(self, net_input: float):
        return 0 if net_input < 0 else 1


class BinarySigmoid(Function):
    def run(self, net_input: float):
        net_input = round_input(net_input)
        return 1 / (1 + np.exp(-net_input))

    def run_differential(self, net_input: float):
        net_input = round_input(net_input)
        s = self.run(net_input)
        return s * (1 - s)


class BipolarSigmoid(Function):
    def __init__(self, factor: float = 1):
        self.factor = factor

    def run(self, net_input: float):
        net_input = round_input(net_input)
        # if math.isnan(net_input):
        #     print('net_input:',   net_input)
        return self.factor*(1 - np.exp(-net_input)) / (1 + np.exp(-net_input))

    def run_differential(self, net_input: float):
        net_input = round_input(net_input)
        s = self.run(net_input)
        return self.factor*(1 - s**2)/2.0

