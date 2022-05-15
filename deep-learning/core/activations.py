import numpy as np
from core.layers import *
from core.optimizers import *
from core.loss import *

class Relu:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    def backward(self, dx):
        self.dinputs = dx.copy()
        self.dinputs[self.inputs <= 0] = 0
    
class Softmax:
    def forward(self, inputs):
        self.inputs = inputs
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities
    
    def backward(self, dx):
        self.dinputs = np.empty_like(dx)
        for index, (single_output, single_dx) in enumerate(zip(self.output, dx)):
            single_output = single_output.reshape(-1, 1)
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            self.dinputs[index] = np.dot(jacobian_matrix, single_dx)