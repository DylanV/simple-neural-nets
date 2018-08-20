"""General purpose network class for building fully connected ANNs."""

from collections import deque
from time import sleep
from typing import List, Tuple

import numpy as np
from tqdm import tqdm

from nn.activations import sigmoid, sigmoid_derivative
from nn.cost import MSE, MSE_derivative
from nn.weights import initailise_weights


class Network:
    """Generic fully connected neural network."""

    def __init__(self, layers: List[int]):
        """
        Parameters
        ----------
        layers : list of int
            Number of neurons per layer including input and output.
        """
        self.weights = []
        self.biases = []
        for i in range(1, len(layers)):
            self.weights.append(initailise_weights((layers[i - 1], layers[i])))
            self.biases.append(np.zeros((1, layers[i])))

    def forward(self, batch: np.ndarray) -> np.ndarray:
        """Do a forward pass of the network with the current weights.

        Parameters
        ----------
        batch : ndarray
            A batch of data with each input a row
            Shape: [batch_size x data_size]

        Returns
        -------
        output_activations : ndarray
            The final layer activations
            Shape: [batch_size x output_layer_size]

        """
        activations = batch
        for weights, biases in zip(self.weights, self.biases):
            activations = sigmoid(np.dot(activations, weights) + biases)
        return activations

    def backward(self, batch, targets) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray]]:
        """Use backpropagation to calculate the gradients for the model parameters.

        Parameters
        ----------
        batch : ndarray
            A batch of data with each input a row
            Shape: [batch_size x data_size]
        targets : ndarray
            The true values for the training data
            Shape: [batch_size x output_layer_size]

        Returns
        -------
        mean_cost, weight_gradients, bias_gradients : tuple of ndarray, list, list
            The mean cost and list containing the gradients for the model parameters
            The list are ordered from input to output layers

        """
        # Do the forward pass with caching
        zs, activations = [], [batch]
        for weights, biases in zip(self.weights, self.biases):
            zs.append(np.dot(activations[-1], weights) + biases)
            activation = sigmoid(zs[-1])
            activations.append(activation)

        # Now do the backwards pass
        cost = np.mean(MSE(activations[-1], targets))

        weight_gradients = deque()
        bias_gradients = deque()

        # Do the output layer error first because it's slightly different to the others
        error = MSE_derivative(activations[-1], targets) * sigmoid_derivative(zs[-1])
        for l in range(2, len(self.weights) + 2):
            # Calculate the gradients from the error
            bias_gradients.appendleft(np.sum(error, 0))
            weight_gradients.appendleft(np.dot(activations[-l].T, error))
            # Backpropagate the error until we get the input layer
            if not l > len(zs):
                error = np.dot(error, self.weights[-l + 1].T) * sigmoid_derivative(zs[-l])

        return cost, list(weight_gradients), list(bias_gradients)

    def fit(self, data, targets, epochs=10, batch_size=5, learning_rate=1e-3):
        """Use mini-batch SGD to learn the weights and biases of the network.

        Parameters
        ----------
        data : ndarray
            The training set
        targets : ndarray
            The expected outputs for the training set
        epochs : int
            The number of epochs to train for
        batch_size : int
            Number of training examples per mini-batch
        learning_rate : float
            The learning rate

        """
        num_samples = data.shape[0]
        batch_starts = np.arange(batch_size, num_samples, batch_size)
        idxes = np.arange(0, num_samples, 1)

        print('EPOCH   |  COST')
        for epoch in range(epochs):
            # Shuffle the ids and split into mini-batches
            np.random.shuffle(idxes)
            batches = np.split(idxes, batch_starts)
            for batch in tqdm(batches, leave=False, desc=f'Epoch {epoch:2d}: '):
                cost, weight_gradients, bias_gradients = self.backward(data[batch], targets[batch])
                # Update the weights
                for i in range(len(self.weights)):
                    self.weights[i] -= learning_rate * (1 / batch_size) * weight_gradients[i]
                    self.biases[i] -= learning_rate * (1 / batch_size) * bias_gradients[i]
            sleep(0.01)  # Tiny sleep so the progress bar updates correctly
            print(f'Epoch {epoch:2d}: {cost:0.2e}')
