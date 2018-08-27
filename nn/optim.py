
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
                for i in range(len(gradients)):
                    delta = learning_rate / batch_size
                    for param, grad in zip(self.parameters[i], gradients[i]):
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

        self.gradient_history = []

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
                # Make sure there is something in the gradient history
                self.gradient_history = gradients if self.gradient_history == [] else self.gradient_history
                # Exponential moving average of the loss
                current_loss = 0.01 * loss + (1 - 0.99) * current_loss if loss != 0 else loss
                # Update the parameters
                delta = learning_rate / batch_size
                for i in range(len(gradients)):
                    for j in range(len(gradients[i])):
                        param = self.parameters[i][j]
                        grad = gradients[i][j]
                        grad_history = self.gradient_history[i][j]
                        # Do any regularization
                        if regulariser == 'L1':
                            param -= delta * regularisation_weight * np.sign(param)
                        elif regulariser == 'L2':
                            param -= regularisation_weight * delta * param
                        # Update the gradient with the momentum
                        grad = momentum * grad_history + (1 - momentum) * grad
                        self.gradient_history[i][j] = grad
                        # Step down the gradient
                        param -= delta * grad
            sleep(0.01)  # Tiny sleep so the progress bar updates correctly
            print(f'Epoch {self.num_epochs:2d}: {current_loss:0.2e}')
