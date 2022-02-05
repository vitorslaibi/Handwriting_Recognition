"""
    Author: Ali Karimiafshar
    Date:   12/09/2021
"""


import numpy as np


class NeuralNetworkLayer:
    def __init__(self, cur_num_neurons:int, next_num_neurons:int=10) -> None:
        """ A Neural Network Layer that creates weights and biases arrays for the Neural Network \
            according to the number of neurons in the current layer and the next layer.

        Args:
            cur_num_neurons (int): Number of neurons on the current layer. Weights matrix columns.
            next_num_neurons (int, optional): Number of neurons on the next layer. Weights matrix rows. Defaults to 10.
        """
        
        # Initial range of randoms weights and biases is [-0.50, 0.50].
        randRange = 0.5
        
        # Seed to constant for consistent outputs, per project criteria.
        np.random.seed(171317)
        
        self.weights = np.random.uniform(-randRange, randRange, size=(next_num_neurons, cur_num_neurons))
        self.biases = np.random.uniform(-randRange, randRange, size=(next_num_neurons, 1))
        self.output = None
        self.output_activated = None
        
        
    def forward_prop(self, inputData:np.array) -> np.array:
        """ Forward propagation. Performs matrix multiplication between weights and input data, then adds biases.

        Args:
            inputData (np.array): Input data or output data from the last neuron layer.

        Returns:
            np.array: The matrix multiplication product of weights and inputData plus the bias
        """
        
        self.output = self.weights.dot(inputData) + self.biases
        return self.output
        
        
    def activation_ReLU(self) -> np.array:
        """ ReLU activation function.

        Returns:
            np.array: The forward propagation output, or zero, whichever is greater.
        """
        
        # Element-wise maximum comparison.
        self.output_activated = np.maximum(0, self.output)
        return self.output_activated
    
    
    def derivative_activation_ReLU(self) -> np.array:
        """ ReLU derivative used for backward propagation. 

        Returns:
            np.array: The derivative of the ReLU activation function, which is 1 when f>0, or 0 otherwise.
        """
        
        # Bool will be converted to 0 or 1 when type casting.
        return self.output_activated > 0
    
    
    def activation_softmax(self) -> np.array:
        """ Softmax activation function.

        Returns:
            np.array: The matrix prediction of the model based on the input data, weights, and biases of the Neural Network Layers. 
        """
        
        # Element-wise exponential function divided by the sum of all exponential functions e^x.
        self.output_activated = np.exp(self.output) / sum(np.exp(self.output))
        return self.output_activated
    
    
    def update_weights_biases(self, dW:np.array, dB:float, alpha:float) -> None:
        """ Updates the weights and biases of the neural network layer based on the back propagation calculations.

        Args:
            dW (np.array): Change in weights.
            dB (float): Change in biases.
            alpha (float): Learning rate.
        """
        
        self.weights -= alpha * dW
        self.biases -= alpha * dB