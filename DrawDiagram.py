import matplotlib.pyplot as plt
import numpy as np


def draw_plot(x, y):

    plt.figure(figsize=(16, 9))
    plt.plot(x, y, color='g')
    plt.xlabel("Noise")
    plt.ylabel("Accuracy")
    plt.xticks(rotation=-90)
    plt.show()