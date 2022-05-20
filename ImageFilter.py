from abc import ABC, abstractmethod
from typing import List, Type

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


class FilterFunction(ABC):

    @abstractmethod
    def run(self, data_frame: pd.DataFrame):
        pass


class ImageFilter:

    def __init__(self, filter_functions: List[FilterFunction]):
        self.filter_functions: List[FilterFunction] = filter_functions

    def run(self, data_frame: pd.DataFrame):
        for func in self.filter_functions:
            data_frame = func.run(data_frame)
        return data_frame


class Mapping(FilterFunction):
    def __init__(self, to_values: [] = [-1, 1]):
        self.to_values = to_values

    def run(self, df: pd.DataFrame):
        span = self.to_values[1]-self.to_values[0]
        df_max = 0
        df_min = 0
        for i in range(df.shape[0]):
            for j in range(df.shape[1]):
                vals = df.iloc[i, j]
                if isinstance(vals, (np.floating, float)):
                    temp_max = vals
                    temp_in = vals
                else:
                    if type(vals) is not np.ndarray:
                        vals = vals.to_numpy()
                    temp_max = np.amax(vals)
                    temp_in = np.amin(vals)
                if temp_max > df_max:
                    df_max = temp_max
                if temp_in < df_min:
                    df_min = temp_in
        vals = None
        for i in range(df.shape[0]):
            for j in range(df.shape[1]):
                df.iloc[i, j] = self.to_values[0] + ((df.iloc[i, j] - df_min)/(df_max - df_min)) * span

        return df


class Convolution(FilterFunction):

    def __init__(self, convolution: np.ndarray, padding: int, striding: int):
        self.convolution = convolution
        self.padding = padding
        self.striding = striding

    def run(self, df: pd.DataFrame):
        new_df = np.ndarray(df.shape, dtype=np.ndarray)
        for i in range(df.shape[0]):
            for j in range(df.shape[1]):
                if type(df.iloc[i, j]) is not np.ndarray:
                    new_df[i, j] = self.__run_convolution(df.iloc[i, j].to_numpy())
                else:
                    new_df[i, j] = self.__run_convolution(df.iloc[i, j])

        return pd.DataFrame(new_df, columns=df.columns)

    def __run_convolution(self, start_data):
        target_data = np.zeros((int((start_data.shape[0]+2*self.padding-self.convolution.shape[0])/self.striding),
                                int((start_data.shape[1]+2*self.padding-self.convolution.shape[1])/self.striding)))
        for i in range(target_data.shape[0]):
            for j in range(target_data.shape[1]):
                start_index = [self.striding*i-self.padding, self.striding*j-self.padding]
                start_index = [start_index[0] if start_index[0] >= 0 else 0,
                               start_index[1] if start_index[1] >= 0 else 0]
                target_part = start_data[start_index[0]:start_index[0]+self.convolution.shape[0],
                                         start_index[1]:start_index[1]+self.convolution.shape[1]]
                target_data[i, j] = np.sum(np.multiply(self.convolution, target_part))

        # fig = plt.figure()
        # fig.add_subplot(1, 2, 1)
        # plt.imshow(start_data)
        # fig.add_subplot(2, 2, 2)
        # plt.imshow(target_data)
        # plt.show()
        return target_data


# need more work
class ParallelFilters(FilterFunction):
    def __init__(self, filters: List[List[FilterFunction]]):
        self.convolution_filters = filters

    def run(self, data_frame: pd.DataFrame):
        filtered_dfs = []
        for i in range(len(self.convolution_filters)):
            new_df = data_frame
            for j in range(len(self.convolution_filters[i])):
                new_df = self.convolution_filters[i][j].run(new_df)
            filtered_dfs.append(new_df)

        return self.flatten_data_frames(filtered_dfs)

    def flatten_data_frames(self, filtered_data: pd.DataFrame):
        size = 0
        for i in range(len(filtered_data[0])):
            size += np.array(filtered_data[0][i]).flatten().shape[0]

        data_array = np.zeros((len(filtered_data), size))
        for i in range(data_array.shape[0]):
            flatten_data = np.ndarray((0,))
            for j in range(len(filtered_data[0])):
                np.concatenate(flatten_data, np.array(filtered_data[i][j]).flatten())
            data_array[i] = flatten_data
        return data_array




class Flatten(FilterFunction):
    def run(self, df: pd.DataFrame):
        filtered_data = []
        for i in range(df.shape[0]):
            for j in range(df.shape[1]):
                arr = df.iloc[i, j]
                if type(arr) is not np.ndarray:
                    arr = arr.to_numpy()
                filtered_data.append(arr.flatten())
        df = pd.DataFrame(data=np.array(filtered_data))
        return df


class PCAFeatureExtraction(FilterFunction):
    def __init__(self, n_component: int = 5):
        self.n_component = n_component

    def run(self, df: pd.DataFrame):
        pca = PCA(n_components=self.n_component)
        pca.fit(df)
        components = np.transpose(pca.components_)
        mapped_data = np.matmul(df, components)

        # for i in range(mapped_data.shape[0]):
        #     fig = plt.figure()
        #     fig.add_subplot(1, 2, 1)
        #     # plt.imshow(start_data)
        #     # fig.add_subplot(2, 2, 2)
        #     plt.imshow(mapped_data.iloc[i].to_numpy().reshape(2, 2))
        #     plt.show()

        mapped_data = pd.DataFrame(mapped_data)

        return mapped_data


class LetterSizeFeatureExtraction(FilterFunction):
    def run(self, df: pd.DataFrame):
        size_df = pd.DataFrame(np.zeros((df.shape[0], 2)))
        for i in range(df.shape[0]):
            for j in range(df.shape[1]):
                image = df.iloc[i, j]
                mean_on_columns = np.mean(image, axis=0)
                col = image.shape[1]
                for k in range(col):
                    if abs(mean_on_columns[k] - image[1, 1]) < 0.02:
                        col = col - 1
                    else:
                        break
                for k in range(col):
                    if abs(mean_on_columns[-k] - image[1, 1]) < 0.02:
                        col = col - 1
                    else:
                        break

                mean_on_rows = np.mean(image, axis=1)
                row = image.shape[0]

                for k in range(row):
                    if abs(mean_on_rows[k] - image[1, 1]) < 0.02:
                        row = row - 1
                    else:
                        break
                for k in range(row):
                    if abs(mean_on_rows[-k] - image[1, 1]) < 0.02:
                        row = row - 1
                    else:
                        break

                size_df.iloc[i] = [row, col]
        return size_df


