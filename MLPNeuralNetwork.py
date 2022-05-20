from random import shuffle
from typing import List

import ActivationFunctions
from ActivationFunctions import *
import pandas as pd

from ImageFilter import ImageFilter, Flatten
from Network import Network
from Utilities import LearningMode, HiddenLayersStructure
from Neurons.NeuronLayers import HiddenNeuronLayer


class MLPNeuralNetwork:

    def __init__(self, train_inputs: pd.DataFrame, train_outputs: pd.DataFrame,
                 hidden_structures: HiddenLayersStructure = HiddenLayersStructure(),
                 input_activation: ActivationFunctions = IdentityFunction(),
                 output_activation: ActivationFunctions = BipolarSigmoid(),
                 predict_activation: ActivationFunctions = BipolarFunction(),
                 input_filter: ImageFilter = None, learning_rate=-1,
                 initial_weights: List[pd.DataFrame] = None,
                 show_logs: bool = False):

        if type(train_outputs) is pd.Series:
            train_outputs = train_outputs.to_frame()
        if type(train_inputs) is pd.Series:
            train_inputs = train_inputs.to_frame()

        self.filter = input_filter
        self.train_inputs = self.alter_filters(train_inputs)
        self.train_outputs = train_outputs
        self.hidden_structures = hidden_structures

        self.network = self.generate_network(predict_activation, input_activation, output_activation,
                                             learning_rate, initial_weights, show_logs)

    def generate_network(self, predict_activation: ActivationFunctions = IdentityFunction(),
                         input_activation: ActivationFunctions = IdentityFunction(),
                         output_activation: ActivationFunctions = BipolarSigmoid(),
                         learning_rate=-1,
                         initial_weights: List[pd.DataFrame] = None,
                         show_logs: bool = False):
        network = Network(self.train_inputs.shape[1], self.train_outputs.shape[1]
                          , input_activation=input_activation
                          , output_activation=output_activation
                          , prediction_activation=predict_activation
                          , initial_weights=initial_weights
                          , show_logs=show_logs)
        for i in range(len(self.hidden_structures)):
            network.add_hidden_layer(HiddenNeuronLayer(network.output_layer.index, self.hidden_structures[i][0],
                                                       self.hidden_structures[i][1], show_logs=show_logs))
        network.generate_connection_layers(learning_rate)
        return network

    def train(self, epochs: int = 100, mode: LearningMode = LearningMode.Batch, show_logs: bool = False):

        if mode == LearningMode.Batch:
            self.__batch_train(epochs)
        elif mode == LearningMode.Incremental:
            self.__sequential_train(epochs)

    def evaluate(self, test_inputs: pd.DataFrame, test_outputs: pd.DataFrame):
        confusion_matrix: pd.DataFrame = None
        if type(test_outputs) is pd.Series:
            value_set = set(test_outputs)
            test_outputs = test_outputs.to_frame()
            confusion_matrix = pd.DataFrame(np.zeros((len(value_set), len(value_set))))
        else:
            confusion_matrix = pd.DataFrame(np.zeros((test_outputs.shape[1], test_outputs.shape[1])))

        if type(test_inputs) is pd.Series:
            test_inputs = test_inputs.to_frame()

        test_inputs = self.alter_filters(test_inputs)

        for i in range(test_inputs.shape[0]):
            output_vector = self.network.feed_test_values(test_inputs.iloc[i])
            if test_outputs.shape[1] == 1:
                n1 = test_outputs.iloc[i][0]-1
                n2 = output_vector[0]-1
                confusion_matrix[n1][n2] += 1
            else:
                for j in range(len(output_vector)):
                    for k in range(test_outputs.shape[1]):
                        if output_vector[j] > 0 and test_outputs.iloc[i, k] > 0:
                            confusion_matrix[k][j] += 1

        print('\nConfusion matrix:\n', confusion_matrix, '\n')

        return np.trace(confusion_matrix) / np.sum(confusion_matrix.values.flatten())

    def __batch_train(self, epochs: int = 100):
        ending_condition = False
        epoch = 0
        print('batch mode ')
        while not ending_condition:
            print('Epoch ', epoch+1)
            for i in range(self.train_inputs.shape[0]):
                self.network.feed_train_values(self.train_inputs.iloc[i], self.train_outputs.iloc[i])
            self.network.update_weights()
            epoch += 1
            ending_condition = epoch > epochs

    def __sequential_train(self, epochs: int = 100, show_logs: bool = False):
        ending_condition = False
        epoch = 0
        print('incremental mode ')
        while not ending_condition:
            print('Epoch ', epoch+1)
            indices = self.train_inputs.index.to_numpy()
            shuffle(indices)
            for i in indices:
                self.network.feed_train_values(self.train_inputs.iloc[i], self.train_outputs.iloc[i])
                self.network.update_weights()

            epoch += 1
            ending_condition = epoch > epochs

    def alter_filters(self, df: pd.DataFrame):
        if self.filter is not None:
            return self.filter.run(df)
        return df

    def has_flatten(self):
        for fil in self.filter.filter_functions:
            fff = isinstance(fil, Flatten)
            if fff:
                return True

        return False