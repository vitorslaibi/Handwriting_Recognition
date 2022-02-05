"""
    Author: Ali Karimiafshar
    Date:   12/09/2021
"""

from NeuralNetworkLayer import NeuralNetworkLayer

import numpy as np
import pickle
from typing import Dict


class DigitRecognizerNeuralNetwork:
    def __init__(self, X:np.array, Y:np.array, SampleSize:int, iterations:int=500, alpha:float=0.10, isTraining=True, learned_wnb:Dict=None) -> None:
        """ The Neural Network designed to classify handwriting digit samples found in the Kaggle dataset. 

        Args:
            X (np.array): The training or testing data. Excludes the label.
            Y (np.array): The data label.
            SampleSize (int): Size of the dataset.
            iterations (int, optional): Number of times the neural network is trained on the training dataset. Defaults to 500.
            alpha (float, optional): Learning rate. Defaults to 0.10.
            isTraining (bool, optional): If True, the network is a training network, otherwise a testing network.
            learned_wnb (Dict, optional): The dictionary of learned weights and biases.
        """
        
        # Create the neuron layers
        self.hiddenLayer = NeuralNetworkLayer(784,10)
        self.outputLayer = NeuralNetworkLayer(10,10)
        
        # Initialize variables
        self.X = X
        self.Y = Y
        self.size = SampleSize
        self.iterations = iterations
        self.alpha = alpha
        
        self.accuracy_progress = []
        
        # train the model if it is a training network, otherwise test cross validation data.
        if isTraining:
            self.gradient_descent()
        else:
            self.test_cross_data(learned_wnb)
            

    def forward_prop(self) -> None:
        """ Calls the appropriate forward propagation and activation functions for each layer.
        """
        
        hiddenLayer_output = self.hiddenLayer.forward_prop(self.X)
        hiddenLayer_activated = self.hiddenLayer.activation_ReLU()
        
        # Outputs of the hidden layer are the inputs to the output layer.
        outputLayer_output = self.outputLayer.forward_prop(hiddenLayer_activated)
        outputLayer_activated = self.outputLayer.activation_softmax()
                

    def get_actual_label(self, Y:np.array) -> np.array:
        """ Generates an array of zeros with a 1 in the index of the actual label of the training data. \
            This is used to calculate the deviation of the network prediction from the actual label.

        Args:
            Y (np.array): The label array of training data

        Returns:
            np.array: array of zeros with only one value of 1 at the index of the label Y.
        """
        
        # Generate array of zeros
        actual_y = np.zeros((Y.size, 10))
        
        # for each row (data), replace the column Y index (label index), with a 1.
        actual_y[[i for i in range(Y.size)], Y] = 1
        
        # Transpose and return the array so that each row corresponds with the value of the label.
        # For example, if the label of the first row is 4, return [0, 0, 0, 0, 1, 0, 0, 0, 0, 0] as the columns of the first row.
        # Then transpose and return.
        return actual_y.T
    
    
    def backward_prop(self) -> None:
        """ Performs the back propagation algorithm to calculate how much weights and biases of each layer need to be changed by.
        """
        
        # Array of zeroes with one 1 at the index of the actual label of the data.
        actual_y = self.get_actual_label(self.Y)
        
        # Determine how the output layer weights and biases need to be updated.
        dZ_outputLayer = self.outputLayer.output_activated - actual_y
        dW_outputLayer = 1 / self.size * dZ_outputLayer.dot(self.hiddenLayer.output_activated.T)
        db_outputLayer = 1 / self.size * np.sum(dZ_outputLayer)
        
        # Determine how the hidden layer weights and biases need to be updated.
        dZ_hiddenLayer = self.outputLayer.weights.T.dot(dZ_outputLayer) * self.hiddenLayer.derivative_activation_ReLU()
        dW_hiddenLayer = 1 / self.size * dZ_hiddenLayer.dot(self.X.T)
        db_hiddenLayer = 1 / self.size * np.sum(dZ_hiddenLayer)
        
        # Updates the weights and biases of each layer.
        self.outputLayer.update_weights_biases(dW_outputLayer, db_outputLayer, self.alpha)
        self.hiddenLayer.update_weights_biases(dW_hiddenLayer, db_hiddenLayer, self.alpha)
        
        
    def get_predictions(self, outputLayer_output:np.array) -> np.array:
        """ calculates the prediction of the neural network based on the maximum of the output layer.

        Args:
            outputLayer_output (np.array): values of the output layer, whose maximum would be the network prediction.

        Returns:
            np.array: index of the maximum value of the output layer, which corresponds with the digit predicted in range [0, 9]. 
        """
        
        return np.argmax(outputLayer_output, 0)


    def get_accuracy(self, predictions:np.array, Y:np.array, numDigitsToShow:int=17) -> float:
        """ Calculates the accuracy of the model.

        Args:
            predictions (np.array): array of the network predictions based on the activation function of the output layer.
            Y (np.array): array of the actual data labels.
            numDigitsToShow (int, optional): Number of first elements of prediction and label arrays to discplay. Defaults to 17.

        Returns:
            float: Accuracy of the model out of 100%.
        """
        
        accuracy = np.sum(predictions == Y) / Y.size
        
        if numDigitsToShow:
            print(f"Network predictions:\t{predictions[:numDigitsToShow]}\nActual labels:\t\t{Y[:numDigitsToShow]}")
        
        return accuracy
    
    
    def save_weights_and_biases(self) -> None:
        """ Pickles the learned weights and biases in a file named learned_wnb.pkl.
        """
        
        with open("learned_wnb.pkl", "wb") as outFile:
            pickle.dump(self.tunes, outFile)

    
    def display_updates(self) -> None:
        """ Displays updated accuracy of the model.
        """
        
        neural_network_output = self.outputLayer.output
        self.predictions = self.get_predictions(neural_network_output)
        
        accuracy = self.get_accuracy(self.predictions, self.Y) * 100
        self.accuracy_progress.append(accuracy)
        
        print(f"Accuracy:\t\t{accuracy:.2f}%")
    
    
    def gradient_descent(self) -> None:
        """ Performs the gradient decent a number of iterations to train the model. \
            Calls forward_prop, and backward_prop, then displays updated accuracy of the model with every 10 iterations.
        """
        
        for i in range(1, self.iterations+1):
            self.forward_prop()
            self.backward_prop()
            
            # For every 10 training iterations, display updates.
            if (i % 10) == 0:
                print(f"iteration: {i}")
                self.display_updates()
        
        # After the training iterations, store the weights and biases in tunes dictionary.
        self.tunes = {"hiddenLayer_weights":self.hiddenLayer.weights,
                      "hiddenLayer_biases":self.hiddenLayer.biases,
                      "outputLayer_weights":self.outputLayer.weights,
                      "outputLayer_biases":self.outputLayer.biases}
        
        self.save_weights_and_biases()
    
    
    def load_weights_and_biases(self, wnb_dict:Dict) -> None:
        """ Loads learned weights and biases into the model for testing unexamined sample data.

        Args:
            wnb_dict (Dict): Dictionary of the weights and biases.
        """
        
        self.hiddenLayer.weights = wnb_dict["hiddenLayer_weights"]
        self.hiddenLayer.biases = wnb_dict["hiddenLayer_biases"]
        self.outputLayer.weights = wnb_dict["outputLayer_weights"]
        self.outputLayer.biases = wnb_dict["outputLayer_biases"]
    
    
    def test_cross_data(self, learned_wnb:Dict) -> None:
        """ Classifies cross validation data based on the learned weights and biases. \
            Then displays the accuracy of the model.

        Args:
            learned_wnb (Dict): learned weights and biases to be loaded into the network.
        """
        
        self.load_weights_and_biases(learned_wnb)
        self.forward_prop()
        self.display_updates()