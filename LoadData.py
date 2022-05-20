import random

import numpy as np
import pandas as pd


def load_data(train_ratio: float = 0.7):
    df = pd.read_csv('iris.data', header=None)

    # output_set = list(set(df[4]))
    # output_df = pd.DataFrame(np.ones((df.shape[0], len(output_set)))*-1, columns=[5, 6, 7])
    # for i in range(len(output_set)):
    #     output_df.iloc[df.loc[df[4] == output_set[i], 4].index, i] = 1
    #     df.loc[df[4] == output_set[i], 4] = i

    output_df = one_hot(df[4])
    output_df.columns = [5, 6, 7]
    df = pd.concat([df, output_df], axis=1)
    # print(df)

    df = df.sample(frac=1, ignore_index=True)

    train_counts = train_ratio * df.shape[0]
    train_df = df.loc[0:train_counts]
    test_df = df.loc[train_counts:df.shape[0]]

    train_inputs = train_df.loc[:, 0:3]
    train_output = train_df.loc[:, 5:7] # train_df[4]
    test_inputs = test_df.loc[:, 0:3]
    test_output = test_df.loc[:, 5:7] # test_df[4]

    return train_inputs, train_output, test_inputs, test_output


def one_hot(input_df):
    output_set = list(set(input_df))
    output_df = pd.DataFrame(np.ones((input_df.shape[0], len(output_set))) * -1)
    for i in range(len(output_set)):
        output_df.iloc[input_df.loc[input_df == output_set[i]].index, i] = 1
        # input_df.loc[input_df == output_set[i]] = i
    return output_df


def add_noise(data, noise: float):
    noisy_inputs = data.copy()
    changes = 0
    for i in range(noisy_inputs.shape[0]):
        for j in range(noisy_inputs.shape[1]):
            if random.random() < noise / 2.0:
                noisy_inputs.iloc[i, j] *= -1
                changes += 1
    print('changes by noise: ', changes)
    return noisy_inputs


def add_noise_to_images(data, noise: float):
    noisy_inputs = data.copy()
    if noise == 0:
        return noisy_inputs
    changes = 0
    for i in range(noisy_inputs.shape[0]):
        img = noisy_inputs[i]
        for j in range(img.shape[0]):
            for k in range(img.shape[1]):
                pixel = img[j][k]
                if pixel < 0.8:
                    if random.random() < noise / 2.0:
                        img[j][k] *= -1
                        changes += 1
    print('changes by noise: ', changes)
    return noisy_inputs