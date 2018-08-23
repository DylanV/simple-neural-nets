

class Sequential:

    def __init__(self, *layers):
        self.layers = [*layers]
        self.parameters = [layer.parameters for layer in self.layers if layer.trainable]

    def forward(self, batch):
        activation = batch
        for layer in self.layers:
            activation = layer.forward(activation)

        return activation

    def backward(self, batch, targets):

        output = self.forward(batch)  # Forward pass
        loss = self.layers[-1].loss(output, targets)  # Calculate loss with output layer
        error = self.layers[-1].backward(output, targets)  # Get starting error from output layer
        for layer in reversed(self.layers[:-1]):
            error = layer.backward(error)  # Backpropagate the error

        return loss, [layer.gradients for layer in self.layers if layer.trainable]
