# from ActivationFunctions import *
# import pandas as pd
# from Network import Network
# from Utilities import LearningMode, HiddenLayersStructure
# from Neurons.NeuronLayers import HiddenNeuronLayer
#
#
# class NeuralNetworkParameters:
#     def __init__(self):
#         # self.layers = []
#         self.hidden_structures: HiddenLayersStructure = HiddenLayersStructure()
#         self.train_inputs: pd.DataFrame = None
#         self.train_outputs: pd.DataFrame = None
#         self.preprocessing_layers = []
#
# class MoreGenericNeuralNetwork:
#
#     def __init__(self, parameters: NeuralNetworkParameters)
#                  # train_inputs: pd.DataFrame, train_outputs: pd.DataFrame,
#                  # hidden_structures: HiddenLayersStructure = HiddenLayersStructure()):
#
#         if type(parameters.train_outputs) is pd.Series:
#             parameters.train_outputs = parameters.train_outputs.to_frame()
#         if type(parameters.train_inputs) is pd.Series:
#             parameters.train_inputs = parameters.train_inputs.to_frame()
#
#         self.parameters = parameters
#         self.train_inputs = parameters.train_inputs
#         self.train_outputs = parameters.train_outputs
#         self.hidden_structures = parameters.hidden_structures
#         self.network = self.generate_network()
#
#     def generate_network(self):
#         network = Network(self.train_inputs.shape[1], self.train_outputs.shape[1]
#                           , input_activation=BipolarSigmoid()
#                           , output_activation=BipolarSigmoid())
#         for i in range(len(self.hidden_structures)):
#             network.add_hidden_layer(HiddenNeuronLayer(self.hidden_structures[i][0], self.hidden_structures[i][1]))
#         network.generate_connection_layers()
#         return network
#
#     def train(self, epochs: int = 100, mode: LearningMode = LearningMode.Batch):
#
#         if mode == LearningMode.Batch:
#             self.__batch_train(epochs)
#         elif mode == LearningMode.Sequential:
#             self.__sequential_train(epochs)
#
#     def evaluate(self, test_inputs: pd.DataFrame, test_outputs: pd.DataFrame):
#         confusion_matrix: pd.DataFrame = None
#         if type(test_outputs) is pd.Series:
#             value_set = set(test_outputs)
#             test_outputs = test_outputs.to_frame()
#             confusion_matrix = pd.DataFrame(np.zeros((len(value_set), len(value_set))))
#         else:
#             confusion_matrix = pd.DataFrame(np.zeros((test_outputs.shape[1], test_outputs.shape[1])))
#
#         if type(test_inputs) is pd.Series:
#             test_inputs = test_inputs.to_frame()
#
#         for i in range(test_inputs.shape[0]):
#             output_vector = self.network.feed_test_values(test_inputs.iloc[i])
#             if test_outputs.shape[1] == 1:
#                 confusion_matrix[test_outputs.iloc[i][0]][output_vector[0]] += 1
#             else:
#                 for j in range(len(output_vector)):
#                     for k in range(test_outputs.shape[1]):
#                         if output_vector[j] > 0 and test_outputs.iloc[i, k] > 0:
#                             confusion_matrix[k][j] += 1
#
#         print('\nConfusion matrix:\n', confusion_matrix, '\n')
#
#         return np.trace(confusion_matrix) / np.sum(confusion_matrix.values.flatten())
#
#     def __batch_train(self, epochs: int = 100):
#         ending_condition = False
#         epoch = 0
#         print('batch mode ')
#         while not ending_condition:
#             print('Epoch ', epoch+1)
#             for i in range(self.train_inputs.shape[0]):
#                 train_input = self.train_inputs[i]
#                 for j in range(len(self.parameters.preprocessing_layers)):
#                     train_input = self.parameters
#                 self.network.feed_train_values(self.train_inputs.iloc[i], self.train_outputs.iloc[i])
#             self.network.update_weights()
#             epoch += 1
#             ending_condition = epoch > epochs
#
#     def __sequential_train(self, epochs: int = 100):
#         ending_condition = False
#         epoch = 0
#         print('incremental mode ')
#         while not ending_condition:
#             print('Epoch ', epoch+1)
#             for i in range(self.train_inputs.shape[0]):
#                 self.network.feed_train_values(self.train_inputs.iloc[i], self.train_outputs.iloc[i])
#                 self.network.update_weights()
#
#             epoch += 1
#             ending_condition = epoch > epochs
#
#
#
