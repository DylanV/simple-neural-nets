
from time import sleep

import numpy as np
from tqdm import tqdm


class SGD:

    def __init__(self, parameters, grad_func):
        self.parameters = parameters
        self.grad_func = grad_func

        self.num_epochs = 0

    def train(self, data, targets, epochs, batch_size, learning_rate, regularisation_weight,
              regulariser='L2'):
        num_samples = data.shape[0]
        batch_starts = np.arange(batch_size, num_samples, batch_size)
        idxes = np.arange(0, num_samples, 1)

        for epoch in range(epochs):
            self.num_epochs += 1
            # Shuffle the ids and split into mini-batches
            np.random.shuffle(idxes)
            batches = np.split(idxes, batch_starts)
            current_loss = 0
            for batch in tqdm(batches, leave=False, desc=f'Epoch {self.num_epochs:2d}: '):
                loss, gradients = self.grad_func(data[batch], targets[batch])
                # Exponential moving average of the loss
                current_loss = 0.01 * loss + (1 - 0.99) * current_loss if loss != 0 else loss
                # Update the parameters
                delta = learning_rate / batch_size
                for param, grad in zip(self.parameters, gradients):
                    # Do any regularization
                    if regulariser == 'L1':
                        param -= delta * regularisation_weight * np.sign(param)
                    elif regulariser == 'L2':
                        param -= regularisation_weight * delta * param
                    # Step down the gradient
                    param -= delta * grad
            sleep(0.01)  # Tiny sleep so the progress bar updates correctly
            print(f'Epoch {self.num_epochs:2d}: {current_loss:0.2e}')


class MomentumSGD:

    def __init__(self, parameters, grad_func):
        self.parameters = parameters
        self.grad_func = grad_func

        self.num_epochs = 0

        self.velocity = [np.zeros(param.shape) for param in self.parameters]

    def train(self, data, targets, epochs, batch_size, learning_rate, regularisation_weight,
              regulariser='L2', momentum=0.9):
        num_samples = data.shape[0]
        batch_starts = np.arange(batch_size, num_samples, batch_size)
        idxes = np.arange(0, num_samples, 1)

        for epoch in range(epochs):
            self.num_epochs += 1
            # Shuffle the ids and split into mini-batches
            np.random.shuffle(idxes)
            batches = np.split(idxes, batch_starts)
            current_loss = 0
            for batch in tqdm(batches, leave=False, desc=f'Epoch {self.num_epochs:2d}: '):
                loss, gradients = self.grad_func(data[batch], targets[batch])
                # Exponential moving average of the loss
                current_loss = 0.01 * loss + (1 - 0.99) * current_loss if loss != 0 else loss
                # Update the parameters
                delta = learning_rate / batch_size
                for param, grad, v in zip(self.parameters, gradients, self.velocity):
                    # Do any regularization
                    if regulariser == 'L1':
                        param -= delta * regularisation_weight * np.sign(param)
                    elif regulariser == 'L2':
                        param -= regularisation_weight * delta * param
                    # Update the velocity
                    v *= momentum
                    v -= (1 - momentum) * delta * grad
                    # Move the parameter with the velocity
                    param += v
            sleep(0.01)  # Tiny sleep so the progress bar updates correctly
            print(f'Epoch {self.num_epochs:2d}: {current_loss:0.2e}')


class NesterovSGD:

    def __init__(self, parameters, grad_func):
        self.parameters = parameters
        self.grad_func = grad_func

        self.num_epochs = 0

        self.velocity = [np.zeros(param.shape) for param in self.parameters]

    def train(self, data, targets, epochs, batch_size, learning_rate, regularisation_weight,
              regulariser='L2', momentum=0.9):
        num_samples = data.shape[0]
        batch_starts = np.arange(batch_size, num_samples, batch_size)
        idxes = np.arange(0, num_samples, 1)

        for epoch in range(epochs):
            self.num_epochs += 1
            # Shuffle the ids and split into mini-batches
            np.random.shuffle(idxes)
            batches = np.split(idxes, batch_starts)
            current_loss = 0
            for batch in tqdm(batches, leave=False, desc=f'Epoch {self.num_epochs:2d}: '):
                loss, gradients = self.grad_func(data[batch], targets[batch])
                # Exponential moving average of the loss
                current_loss = 0.01 * loss + (1 - 0.99) * current_loss if loss != 0 else loss
                # Update the parameters
                delta = learning_rate / batch_size
                for param, grad, v in zip(self.parameters, gradients, self.velocity):
                    # Save the current velocity
                    v_old = v.copy()
                    # Do any regularization
                    if regulariser == 'L1':
                        param -= delta * regularisation_weight * np.sign(param)
                    elif regulariser == 'L2':
                        param -= regularisation_weight * delta * param
                    # Update the velocity
                    v *= momentum
                    v -= (1 - momentum) * delta * grad
                    # Move the parameter with the new velocity
                    param += (-momentum * v_old) + (1 + momentum) * v
            sleep(0.01)  # Tiny sleep so the progress bar updates correctly
            print(f'Epoch {self.num_epochs:2d}: {current_loss:0.2e}')


class Adagrad:

    def __init__(self, parameters, grad_func):
        self.parameters = parameters
        self.grad_func = grad_func
        self.grad_cache = [np.zeros(param.shape) for param in self.parameters]

        self.num_epochs = 0

    def train(self, data, targets, epochs, batch_size, learning_rate, regularisation_weight,
              regulariser='L2'):
        num_samples = data.shape[0]
        batch_starts = np.arange(batch_size, num_samples, batch_size)
        idxes = np.arange(0, num_samples, 1)

        for epoch in range(epochs):
            self.num_epochs += 1
            # Shuffle the ids and split into mini-batches
            np.random.shuffle(idxes)
            batches = np.split(idxes, batch_starts)
            current_loss = 0
            for batch in tqdm(batches, leave=False, desc=f'Epoch {self.num_epochs:2d}: '):
                loss, gradients = self.grad_func(data[batch], targets[batch])
                # Exponential moving average of the loss
                current_loss = 0.01 * loss + (1 - 0.99) * current_loss if loss != 0 else loss
                # Update the parameters
                delta = learning_rate / batch_size
                for param, grad, cache in zip(self.parameters, gradients, self.grad_cache):
                    # Do any regularization
                    if regulariser == 'L1':
                        param -= delta * regularisation_weight * np.sign(param)
                    elif regulariser == 'L2':
                        param -= regularisation_weight * delta * param
                    cache += grad ** 2
                    # Step down the gradient
                    param -= delta * (grad / (np.sqrt(cache) + 1e-8))
            sleep(0.01)  # Tiny sleep so the progress bar updates correctly
            print(f'Epoch {self.num_epochs:2d}: {current_loss:0.2e}')

