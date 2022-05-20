import math
from typing import List

import numpy as np

from ActivationFunctions import Function, IdentityFunction, ConstantFunction, BipolarFunction
from abc import ABC, abstractmethod


class IConnection(ABC):

    @abstractmethod
    def update_weight(self, delta_w):
        pass

    @abstractmethod
    def feed_to_next(self, value: float):
        pass

    @abstractmethod
    def update_delta(self, delta_w):
        pass


class Neuron(ABC):

    def __init__(self, index: int, layer_index: int, activation: Function = IdentityFunction(), show_logs: bool = False):
        self.index: int = index
        self.layer_index = layer_index
        self.activation = activation
        self.input_connections: List[IConnection] = []
        self.output_connections: List[IConnection] = []
        self.dropped_output_connections: List[IConnection] = []
        self.dropped_input_connections: List[IConnection] = []
        self.net_input = 0.0
        self.output = 0.0
        self.delta = 0.0
        self.delta_in = 0.0
        self.show_logs = show_logs

    def add_input_connection(self, connection: IConnection):
        if connection not in self.input_connections:
            self.input_connections.append(connection)

    def add_output_connection(self, connection: IConnection):
        if connection not in self.output_connections:
            self.output_connections.append(connection)

    def update_output(self):
        if self.activation is not None:
            self.output = self.activation.run(self.net_input)

    def broad_cast(self):
        for i in range(len(self.output_connections)):
            self.output_connections[i].feed_to_next(self.output)

    def drop_output_connection(self, connection: IConnection):
        if connection in self.output_connections:
            self.dropped_output_connections.append(connection)
            self.output_connections.remove(connection)

    def drop_input_connection(self, connection: IConnection):
        if connection in self.input_connections:
            self.dropped_input_connections.append(connection)
            self.input_connections.remove(connection)

    def feed_value(self, value: float):
        self.net_input += value

    def feed_delta_in(self, delta_in):
        # if delta_in > 0:
        #     print(delta_in)
        self.delta_in += delta_in

    def update_delta(self):
        self.delta = self.delta_in * self.activation.run_differential(self.net_input)
        self.delta_in = 0

    def back_propagate(self):
        if self.input_connections is None:
            return
        for i in range(len(self.input_connections)):
            self.input_connections[i].update_delta(self.delta)

    def reset_temp_iter_values(self):
        self.delta_in = 0
        self.net_input = 0


class InputNeuron(Neuron):

    def __init__(self, index: int, layer_index: int, activation: Function = IdentityFunction(), show_logs: bool = False):
        super().__init__(index, layer_index, activation, show_logs=show_logs)

    def set_value(self, input_value: float):
        self.net_input = input_value
        self.output = self.activation.run(self.net_input)

    def feed_delta_in(self, delta_in):
        return

    def update_delta(self):
        return


class OutputNeuron(Neuron):

    def __init__(self, index: int, layer_index: int, activation: Function = IdentityFunction(),
                 prediction_activation: Function = BipolarFunction(), show_logs: bool = False):
        super().__init__(index, layer_index, activation, show_logs=show_logs)
        self.target_output = 0
        self.error = 0
        self.prediction_activation = prediction_activation

    def set_target_value(self, target_output: float):
        self.target_output = target_output

    def generate_train_outputs(self):
        if self.show_logs:
            print('self.net_input: ', "%.3f"%self.net_input, '   self.target_output: ', self.target_output)
        self.output = self.activation.run(self.net_input)
        self.error = self.target_output - self.output
        differential = self.activation.run_differential(self.net_input)
        self.delta = -self.error * differential
        # if self.delta
        if self.show_logs:
            print(f'delta o - {self.index}: {"%.3f"%self.delta}')
        # if self.target_output > 0:
        #     print('error: ', self.error)

    def generate_test_outputs(self):
        # self.output = self.prediction_activation.run(self.net_input)
        return self.net_input


class BiasNeuron(Neuron):

    def __init__(self, index: int, layer_index: int, show_logs: bool = False):
        self.show_logs = show_logs
        self.output_connections: List[IConnection] = []
        self.output = 1.0
        self.index = index
        self.layer_index = layer_index
        self.activation = None
        self.dropped_output_connections: List[IConnection] = []
        self.dropped_input_connections: List[IConnection] = []

    def add_output_connection(self, connection: IConnection):
        if connection not in self.output_connections:
            self.output_connections.append(connection)

    def broad_cast(self):
        for i in range(len(self.output_connections)):
            self.output_connections[i].feed_to_next(self.output)

    def update_delta(self):
        return

    def back_propagate(self):
        return

    def feed_delta_in(self, delta_in):
        return
