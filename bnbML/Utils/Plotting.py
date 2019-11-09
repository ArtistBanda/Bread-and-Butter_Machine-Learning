import numpy as np
import matplotlib.pyplot as plt


def plotLossGraph(loss_history, iter_count):
    plt.figure()
    plt.plot(np.arange(iter_count), loss_history)
    plt.xlabel('Number of Iterations')
    plt.ylabel('Loss')
    plt.show()
