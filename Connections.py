import random
from typing import List

import pandas as pd

from Neurons.NeuronLayers import NeuronLayer, OutputNeuronLayer
from Neurons.Neurons import Neuron, IConnection


class Connection(IConnection):

    def __init__(self, input_neuron: Neuron, output_neuron: Neuron,
                 initial_weight: float = 0.1, learning_rate: float = 0.1, show_logs: bool = False):
        self.input_neuron: Neuron = input_neuron
        input_neuron.add_output_connection(self)
        self.output_neuron = output_neuron
        output_neuron.add_input_connection(self)
        self.weight = initial_weight
        self.learning_rate = learning_rate
        self.delta_w = 0
        self.show_logs = show_logs

    def feed_to_next(self, value: float):
        self.output_neuron.feed_value(value * self.weight)

    def update_delta(self, delta):
        self.delta_w += self.learning_rate * self.input_neuron.output * delta
        self.input_neuron.feed_delta_in(self.weight * delta)

    def update_weight(self):
        if self.show_logs:
            print(f'before=> layer = {self.input_neuron.layer_index+1}: W_{self.input_neuron.index+1},{self.output_neuron.index+1} ='
                  f' {self.weight}, delta: {"%.3f" % self.delta_w}')
        self.weight -= self.delta_w
        if self.show_logs:
            print(f'after => layer = {self.input_neuron.layer_index+1}: W_{self.input_neuron.index+1},{self.output_neuron.index+1} ='
                  f' {"%.3f"%self.weight}')
        self.delta_w = 0

        # if abs(self.weight < 0.02):
        #     self.input_neuron.drop_output_connection(self)
        #     self.output_neuron.drop_input_connection(self)


class ConnectionLayer:

    def __init__(self, input_layer: NeuronLayer, output_layer: NeuronLayer, learning_rate=-1,
                 initial_weights: pd.DataFrame = None, show_logs: bool = False):
        self.connections: List[Connection] = []
        self.dropped_connections: List[Connection] = []
        self.input_layer = input_layer
        self.output_layer = output_layer
        out_size = output_layer.size
        if not isinstance(self.output_layer, OutputNeuronLayer):
            out_size = output_layer.size - 1
        if learning_rate == -1:
            learning_rate = 0.2 / input_layer.size
        for i in range(input_layer.size):
            for j in range(out_size):
                wij = random.uniform(0.05, 0.25) if initial_weights is None else initial_weights.iloc[i, j]
                self.connections.append(Connection(input_layer.neurons[i], output_layer.neurons[j], wij, learning_rate,
                                                   show_logs))

    def update_weights(self):
        for i in range(len(self.connections)):
            self.connections[i].update_weight()
            # if self.connections[i].weight < 0.02:
            #     self.dropped_connections.append(self.connections[i])

        # for connection in self.dropped_connections:
        #     if connection in self.connections:
        #         self.connections.remove(connection)
    #
    # def drop_connection(self, connection: Connection):
    #     if connection in self.connections:
    #         self.dropped_connections.append(connection)
    #         self.connections.remove(connection)



