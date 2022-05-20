from abc import ABC, abstractmethod
from typing import List

import numpy as np

from ActivationFunctions import Function, IdentityFunction, ConstantFunction, BipolarFunction
from Neurons.Neurons import Neuron, InputNeuron, OutputNeuron, BiasNeuron


class NeuronLayer(ABC):

    def __init__(self, index: int, size: int, activation: Function = IdentityFunction(),
                 next_layer=None, previous_layer=None, show_logs: bool = False):
        self.index = index
        self.activation = activation
        self.size = size
        self.next_layer: NeuronLayer = next_layer
        self.previous_layer: NeuronLayer = previous_layer
        self.neurons = []
        self.show_logs = show_logs
        self._generate_neurons()

    @abstractmethod
    def _generate_neurons(self):
        pass

    @abstractmethod
    def append(self, neuron: Neuron):
        pass

    def feed_forward(self):

        for i in range(len(self.neurons)):
            self.neurons[i].update_output()
        if self.show_logs:
            for i in range(len(self.neurons)):
                print(f'Layer {self.index}=> neuron[{i}] input = {self.neurons[i].net_input}')
                print(f'Layer {self.index}=> neuron[{i}] output = {self.neurons[i].output}')

        if self.next_layer is None:
            return

        self.next_layer.reset_net_inputs()
        for i in range(len(self.neurons)):
            self.neurons[i].broad_cast()
        self.next_layer.feed_forward()

    def reset_net_inputs(self):
        for i in range(len(self.neurons)):
            self.neurons[i].net_input = 0

    def update_delta(self):
        for i in range(len(self.neurons)):
            self.neurons[i].update_delta()

    def back_propagate(self):
        for i in range(len(self.neurons)):
            self.neurons[i].back_propagate()
        if self.previous_layer is None:
            return
        self.previous_layer.update_delta()
        self.previous_layer.back_propagate()

    def reset_temp_iter_values(self):
        for i in range(len(self.neurons)):
            self.neurons[i].reset_temp_iter_values()
        if self.next_layer is None:
            return
        self.next_layer.reset_temp_iter_values()


class HiddenNeuronLayer(NeuronLayer):

    def __init__(self, index: int, size: int, activation: Function = IdentityFunction(),
                 next_layer: NeuronLayer = None, previous_layer: NeuronLayer = None, show_logs: bool = False):
        super().__init__(index, size, activation, next_layer, previous_layer, show_logs=show_logs)

    def _generate_neurons(self):
        for i in range(self.size-1):
            self.neurons.append(Neuron(i, self.index, self.activation, show_logs=self.show_logs))
        self.neurons.append(BiasNeuron(self.size - 1, self.index, show_logs=self.show_logs))

    def append(self, neuron: Neuron):
        self.neurons.append(neuron)
        self.size += 1


class InputNeuronLayer(NeuronLayer):

    def __init__(self, index: int, size: int, activation: Function = IdentityFunction(),
                 next_layer: NeuronLayer = None, previous_layer: NeuronLayer = None, show_logs: bool = False):
        super().__init__(index, size, activation, next_layer, previous_layer, show_logs=show_logs)

    def _generate_neurons(self):
        for i in range(self.size - 1):
            self.neurons.append(InputNeuron(i, self.index, self.activation, show_logs=self.show_logs))
        self.neurons.append(BiasNeuron(self.size - 1, self.index, show_logs=self.show_logs))

    def append(self, neuron: InputNeuron):
        self.neurons.append(neuron)
        self.size += 1

    def feed_values(self, input_samples: np.ndarray):
        self.reset_temp_iter_values()
        for i in range(input_samples.shape[0]):
            self.neurons[i].feed_value(input_samples.iloc[i])
        forward = self.feed_forward()


class OutputNeuronLayer(NeuronLayer):

    def __init__(self, index: int, size: int, activation: Function = IdentityFunction(),
                 next_layer: NeuronLayer = None, previous_layer: NeuronLayer = None,
                 prediction_activation: Function = BipolarFunction(), show_logs: bool = False):
        self.prediction_activation = prediction_activation
        super().__init__(index, size, activation, next_layer, previous_layer, show_logs=show_logs)
        self.output = 0

    def _generate_neurons(self):
        for i in range(self.size):
            self.neurons.append(OutputNeuron(i, self.index, self.activation,
                                             prediction_activation=self.prediction_activation, show_logs=self.show_logs))

    def append(self, neuron: OutputNeuron):
        self.neurons.append(neuron)
        self.size += 1

    def feed_forward(self):
        super().feed_forward()
        # self.generate_outputs()

    def generate_train_outputs(self, output_samples: np.ndarray):
        for i in range(len(self.neurons)):
            self.neurons[i].set_target_value(output_samples.iloc[i])
        for i in range(len(self.neurons)):
            self.neurons[i].generate_train_outputs()

    def generate_test_outputs(self):
        output_vector = []
        for i in range(len(self.neurons)):
            output_vector.append(self.neurons[i].generate_test_outputs())
        if len(self.neurons) > 1:
            max_val = max(output_vector)
            output_vector = [1 if x == max_val else -1 for x in output_vector]
        else:
            output_vector = [np.rint(x) for x in output_vector]
        return output_vector
