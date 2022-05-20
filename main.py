# This is a sample Python script.
from typing import List

import numpy as np
import pandas as pd

from ActivationFunctions import BipolarSigmoid, ReluFunction, BipolarFunction, IdentityFunction, BinarySigmoid, \
    BinaryFunction
from ImageFilter import ImageFilter, Convolution, Flatten, Mapping, PCAFeatureExtraction, LetterSizeFeatureExtraction
from MLPNeuralNetwork import MLPNeuralNetwork
from SKLearnMLP import classify_by_sklearn_mlp
from Utilities import HiddenLayersStructure, LearningMode, make_gaussian
from LoadData import load_data, add_noise
from LoadImages import read_image_data_text

from DrawDiagram import draw_plot
from KerasMLP import UseKerasToLearn


def mlp_for_project_2(use_iris: bool = False, learning_rate=-1, hidden_layer_neurons: int = 2, epochs: int = 10,
                      noise_step: float = 0.2, show_logs: bool = False):

    initial_weights: List[pd.DataFrame] = []
    if use_iris:
        train_inputs, train_output, test_inputs, test_output = load_data()
        initial_weights = None
    else:
        train_inputs = pd.DataFrame([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]])
        initial_weights.append(pd.DataFrame([[0.70, 0.21], [0.18, 0.11], [0.38, 0.44], [0.22, 0.57]]))
        initial_weights.append(pd.DataFrame([[0.77, 0.86], [0.17, 0.29], [0.09, 0.81]]))
        train_output = pd.DataFrame([[1, 1], [-1, -1], [-1, 1], [-1, 1]])
        test_inputs = train_inputs
        test_output = train_output

    accuracies = []
    x_values = np.arange(0, 1, noise_step)
    for noise in x_values:
        print('\nNoise =', noise, ' :\n')
        # train_inputs_noisy = add_noise(train_inputs, noise)
        train_output_noisy = add_noise(train_output, noise)
        # test_inputs_noisy = add_noise(test_inputs, noise)
        test_output_noisy = add_noise(test_output, noise)

        hidden_layer_structure = HiddenLayersStructure()
        hidden_layer_structure.add_layer(hidden_layer_neurons, BipolarSigmoid())
        # hidden_layer_structure.add_layer(128, BipolarSigmoid())

        mlp = MLPNeuralNetwork(train_inputs, train_output_noisy, hidden_layer_structure,
                               predict_activation=BipolarFunction(), learning_rate=learning_rate,
                               initial_weights=initial_weights, show_logs=show_logs)
        mlp.train(epochs, mode=LearningMode.Incremental)
        accuracy = mlp.evaluate(train_inputs, train_output_noisy)
        print('\ntrain accuracy1: ', accuracy, '\n')
        accuracy = mlp.evaluate(test_inputs, test_output_noisy)
        print('\ntest accuracy2: ', accuracy, '\n')
        accuracies.append(accuracy)
    # mlp.evaluate(test_inputs, test_output)
    draw_plot(x_values, accuracies)


def mlp_for_project_3(learning_rate=-1, hidden_layer_neurons: [] = [8, 8], epochs: int = 10, noise_step: float = 0.2
                      , show_logs: bool = False):
    train_input, train_output, test_input, test_output = read_image_data_text('farsi7.txt', 95, 95, image_size=40)

    conv_size = 10
    convolution_filter = make_gaussian(-1, 1, conv_size)
    convolution_filter2 = np.array([[1, 0, 1], [1, 0, 1], [1, 0, 1]])
    # convolution_filter = np.ones((conv_size, conv_size))
    filters = ImageFilter(5, 0, filter_functions=[
                                                    # LetterSizeFeatureExtraction(),
                                                  Convolution(convolution_filter, 0, conv_size)
                                                  # , Mapping([1, 0])
                                                  # , Convolution(convolution_filter2, 0, conv_size)
                                                  , Flatten()
                                                  # , PCAFeatureExtraction(n_component=8)
                                                  , Mapping([-1, 1])
                                                  ])
    hidden_layer_structure = HiddenLayersStructure()
    for i in range(len(hidden_layer_neurons)):
        hidden_layer_structure.add_layer(hidden_layer_neurons[i], BipolarSigmoid())

    mlp = MLPNeuralNetwork(train_input, train_output, hidden_layer_structure,
                           input_activation=IdentityFunction(),
                           output_activation=BipolarSigmoid(),
                           predict_activation=BipolarFunction(),
                           input_filter=filters, learning_rate=learning_rate, show_logs=show_logs)
    mlp.train(epochs, mode=LearningMode.Batch)
    accuracy = mlp.evaluate(train_input, train_output)


    # accuracy = classify_by_sklearn_mlp(train_input, train_output, test_input, test_output, filters)
    print('\n accuracy= ', accuracy, '\n')


def mlp_for_project_3_with_keras(learning_rate=-1, hidden_layer_neurons: [] = [8, 8], epochs: int = 10,
                                 noise_step: float = 0.2):
    train_input, train_output, test_input, test_output = read_image_data_text('farsi7.txt', 95, 95, image_size=40)

    conv_size = 10
    convolution_filter = make_gaussian(-1, 1, conv_size)
    convolution_filter2 = np.array([[1, 0, 1], [1, 0, 1], [1, 0, 1]])
    # convolution_filter = np.ones((conv_size, conv_size))
    filters = ImageFilter(5, 0, filter_functions=[
                                                  Convolution(convolution_filter, 0, conv_size)
                                                  # , Mapping([1, 0])
                                                  # , Convolution(convolution_filter2, 0, conv_size)
                                                  , Flatten()
                                                  # , PCAFeatureExtraction(n_component=16)
                                                  , Mapping([-1, 1])
                                                  ])
    mean_accuracy = 0
    for i in range(5):
        mlp = UseKerasToLearn(train_input, train_output, filters)
        mlp.train(epochs, mode=LearningMode.Incremental)
        accuracy = mlp.evaluate(train_input, train_output)
        mean_accuracy += accuracy
    mean_accuracy = mean_accuracy / 5

    # accuracy = classify_by_sklearn_mlp(train_input, train_output, test_input, test_output, filters)
    print('\n train accuracy= ', mean_accuracy, '\n')
    accuracy = mlp.evaluate(test_input, test_output)
    print('\n test accuracy= ', mean_accuracy, '\n')


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # mlp_for_project_2(use_iris=True, hidden_layer_neurons=8, epochs=50,
    #                   noise_step=0.2, learning_rate=0.2, show_logs=False)
    # mlp_for_project_3(hidden_layer_neurons=[32, 32, 32, 24], epochs=400, noise_step=0.5, show_logs=False)
    mlp_for_project_3_with_keras(epochs=1000, noise_step=0.5)

