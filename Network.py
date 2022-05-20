from typing import List

import pandas as pd

from Connections import ConnectionLayer
from ActivationFunctions import *
from Neurons.NeuronLayers import HiddenNeuronLayer, InputNeuronLayer, OutputNeuronLayer


class Network:

    def __init__(self, input_size: int, output_size: int,
                 input_activation: Function = IdentityFunction(),
                 output_activation: Function = IdentityFunction(),
                 prediction_activation: Function = BipolarFunction(),
                 initial_weights: List[pd.DataFrame] = None,
                 show_logs: bool = False):
        self.input_layer: InputNeuronLayer = InputNeuronLayer(0, input_size+1, input_activation, show_logs=show_logs)
        self.output_layer: OutputNeuronLayer = OutputNeuronLayer(1, output_size, output_activation,
                                                                 prediction_activation=prediction_activation,
                                                                 show_logs=show_logs)
        self.hidden_layers: List[HiddenNeuronLayer] = []
        self.connection_layers: List[ConnectionLayer] = []
        self.initial_weights = initial_weights
        self.show_logs = show_logs

    def add_hidden_layer(self, neuron_layer: HiddenNeuronLayer):
        self.hidden_layers.append(neuron_layer)
        self.output_layer.index = self.output_layer.index+1

    # def add_hidden_layer(self, size: int, activation: ActivationFunction = IdentityFunction()):
    #     self.hidden_layers.append(NeuronLayer(size, activation))

    def generate_connection_layers(self, learning_rate=-1, initial_weights: pd.DataFrame = None):
        pre_layer = self.input_layer
        for i in range(len(self.hidden_layers)):
            weights = None if self.initial_weights is None else self.initial_weights[i]
            self.connection_layers.append(ConnectionLayer(pre_layer, self.hidden_layers[i], learning_rate, weights,
                                                          show_logs=self.show_logs))
            pre_layer.next_layer = self.hidden_layers[i]
            self.hidden_layers[i].previous_layer = pre_layer
            pre_layer = self.hidden_layers[i]
        pre_layer.next_layer = self.output_layer
        self.output_layer.previous_layer = pre_layer
        weights = None if self.initial_weights is None else self.initial_weights[-1]
        self.connection_layers.append(ConnectionLayer(pre_layer, self.output_layer, learning_rate, weights,
                                                      show_logs=self.show_logs))

    def feed_train_values(self, input_sample: np.ndarray, output_sample: np.ndarray):
        self.input_layer.feed_values(input_sample)
        self.output_layer.generate_train_outputs(output_sample)
        self.output_layer.back_propagate()

    def feed_test_values(self, input_sample: np.ndarray):
        self.input_layer.feed_values(input_sample)
        output_vector = self.output_layer.generate_test_outputs()
        return output_vector

    def update_weights(self):
        for i in range(len(self.connection_layers)):
            self.connection_layers[i].update_weights()
