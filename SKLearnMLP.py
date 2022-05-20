import math

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt

from ImageFilter import ImageFilter


def alter_filters(df: pd.DataFrame, filters: ImageFilter):
    if filters is not None:
        return filters.run(df)
    return df


def classify_by_sklearn_mlp(train_input: pd.DataFrame, train_output: pd.DataFrame,
                            test_inputs: pd.DataFrame, test_outputs: pd.DataFrame, filters: ImageFilter):

    if type(train_output) is pd.Series:
        train_output = train_output.to_frame()
    if type(train_input) is pd.Series:
        train_input = train_input.to_frame()
    if type(test_inputs) is pd.Series:
        test_inputs = test_inputs.to_frame()
    if type(test_outputs) is pd.Series:
        test_outputs = test_outputs.to_frame()

    train_input = alter_filters(train_input, filters)
    test_inputs = alter_filters(test_inputs, filters)
    print(train_input.shape)
    print(train_output.shape)

    # sqrt_size = int(math.sqrt(train_input.shape[1]))
    # for i in range(train_input.shape[0]):
    #     plt.imshow(train_input.iloc[i, :].to_numpy().reshape(sqrt_size, sqrt_size))
    #     plt.show()


    clf = MLPClassifier(max_iter=5000, activation='relu', hidden_layer_sizes=(64, 64)).fit(train_input, train_output) #random_state=1,


    confusion_matrix: pd.DataFrame = None
    if type(test_outputs) is pd.Series:
        value_set = set(test_outputs)
        test_outputs = test_outputs.to_frame()
        confusion_matrix = pd.DataFrame(np.zeros((len(value_set), len(value_set))))
    else:
        confusion_matrix = pd.DataFrame(np.zeros((test_outputs.shape[1], test_outputs.shape[1])))

    if type(test_inputs) is pd.Series:
        test_inputs = test_inputs.to_frame()
    predictions = clf.predict(test_inputs)
    for i in range(test_inputs.shape[0]):
        output_vector = predictions[i]
        if test_outputs.shape[1] == 1:
            n1 = test_outputs.iloc[i][0] - 1
            n2 = output_vector[0] - 1
            confusion_matrix[n1][n2] += 1
        else:
            for j in range(len(output_vector)):
                for k in range(test_outputs.shape[1]):
                    if output_vector[j] > 0 and test_outputs.iloc[i, k] > 0:
                        confusion_matrix[k][j] += 1

    print('\nConfusion matrix:\n', confusion_matrix, '\n')

    return np.trace(confusion_matrix) / np.sum(confusion_matrix.values.flatten())


