# simple-neural-nets
Basic implementation of neural network concepts. Focus on readability over performance. 

## Setup
Run `conda env create environment.yml` in the top level directory
then `activate nn` or `source activate nn` to activate the environment

## Currently Implemented
### General
* Generic fully connected networks with arbitrary layers.
* Finite difference gradient checker.
* Unit tests, most individual components are unit tested.
### Activations
* Sigmoid (logistic)
* Tanh
* Rectified Linear Unit (ReLU) 
### Cost/Loss Functions
* Mean squared error (MSE)
### Optimization
* Mini-batch Stochastic Gradient Descent (SGD)
* SGD with momentum
### Regularisation
* L1
* L2
* Dropout
### Weight initialisation
* Random
* Xavier
* He
### Data loaders
* Automatic downloading and loading of MNIST data
