import numpy as np
from core.activations import *
from core.loss import *
from core.optimizers import *

class Linear:
    def __init__(self, n_inputs, n_neurons):
        super().__init__()
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dx):
        self.dweights = np.dot(self.inputs.T, dx)
        self.dbiases = np.sum(dx, axis=0, keepdims=True)
        self.dinputs = np.dot(dx, self.weights.T)