"""
helper functions
"""

import torch
import numpy as np
import matplotlib.pyplot as plt

def transformMatrixToTensor(x):
    """
    change numpy matrix / array to tensor
    """
    return torch.stack([torch.Tensor(i) for i in x])

def flatten(x):
    N = x.shape[0]
    return x.view(N, -1)

def plotLoss(loss, savePath=None):

    fig = plt.figure()
    plt.semilogy(range(len(loss)), loss)
    plt.show()
    if savePath:
        plt.savefig(savePath, bbox_inches="tight")
        print("saved to: %s" %(savePath))

    return fig

def plotPrices(model, predictions, savePath=None):

    fig = plt.figure()
    plt.plot(model.y[model.train_size: ], label="True")
    plt.plot(predictions, label="Predicted", linestyle="--")
    plt.legend(loc="upuper_left")
    plt.title("Test Results")
    plt.show()
    if savePath:
        plt.savefig(savePath, bbox_inches="tight")
        print("saved to: %s" %(savePath))

    return fig