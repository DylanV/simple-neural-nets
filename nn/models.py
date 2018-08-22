

class Sequential:

    def __init__(self, *layers):
        self.layers = [*layers]

    def forward(self, batch):
        activation = batch
        for layer in self.layers:
            activation = layer.forward(activation)

        return activation

    def backward(self, batch, targets):

        output = self.forward(batch)
        error = self.layers[-1].backward(output, targets)
        for layer in reversed(self.layers[:-1]):
            error = layer.backward(error)
