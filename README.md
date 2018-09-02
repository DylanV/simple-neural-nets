# simple-neural-nets
Basic implementation of neural network concepts. Focus on readability over performance. 

## Setup
Run `conda env create environment.yml` in the top level directory
then `activate nn` or `source activate nn` to activate the environment

## Simple MNIST example network
```python
import numpy as np

from nn.data_loaders import MNISTLoader
from nn.models import Sequential
from nn.activations import Linear, Sigmoid
from nn.cost import SoftmaxCrossEntropy
from nn.optim import NesterovSGD

# Load MNIST
mloader = MNISTLoader()
X, y = mloader.get_training_set()

# Setup a simple 1 hidden layer network
net = Sequential(Linear(28 * 28, 100), Sigmoid(), Linear(100, 10), Sigmoid(), SoftmaxCrossEntropy())

# Train the model
optimiser = NesterovSGD(net.parameters, net.backward)
optimiser.train(X, y, epochs=5, batch_size=10, learning_rate=1, regularisation_weight=1e-3)

# Grab the test set
X_test, y_test = mloader.get_test_set()
# Inference
y_pred = net.forward(X_test)
# Check our accuracy
correct = np.count_nonzero(np.argmax(y_pred, 1) == np.argmax(y_test, 1))
print(correct / X_test.shape[0])
```
Which gives us more the 90% accuracy

## Currently Implemented
### General
* Generic fully connected networks with arbitrary layers.
* Finite difference gradient checker.
* Unit tests, most individual components are unit tested.
### Optimization
* Mini-batch Stochastic Gradient Descent (SGD)
* SGD with momentum
* SGD with Nesterov momentum
* Adagrad
* RMSProp
* Adam
### Activations
* Sigmoid (logistic)
* Tanh
* Rectified Linear Unit (ReLU) 
### Cost/Loss Functions
* Mean squared error (MSE)
* Softmax / Cross entropy
### Regularisation
* L1 / L2 (Weight decay)
* Dropout
### Weight initialisation
* Xavier
* He
### Data loaders
* Automatic downloading and loading of MNIST data
