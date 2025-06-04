import random
from microautograd.engine import Value

def apply_activation(x: Value, name: str) -> Value:
    """Apply the selected activation function to the Value."""
    if name == "tanh":
        return x.tanh()
    elif name == "relu":
        return x.relu()
    elif name == "sigmoid":
        return x.sigmoid()
    elif name == "linear":
        return x  # identity
    else:
        raise ValueError(f"Unsupported activation function: '{name}'")

class Neuron:
    """A single neuron with a list of weights, a bias, and an activation function."""

    def __init__(self, nin, activation="tanh"):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))
        self.activation = activation

    def __call__(self, x):
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return apply_activation(act, self.activation)

    def parameters(self):
        return self.w + [self.b]

    def __repr__(self):
        return f"Neuron({len(self.w)}, activation='{self.activation}')"

class Layer:
    """A layer of fully-connected neurons."""

    def __init__(self, nin, nout, activation="tanh"):
        self.neurons = [Neuron(nin, activation) for _ in range(nout)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]

    def __repr__(self):
        return f"Layer(nin={len(self.neurons[0].w)}, nout={len(self.neurons)}, activation='{self.neurons[0].activation}')"

class MLP:
    """A multi-layer perceptron model composed of multiple layers."""

    def __init__(self, nin, layer_configs):
        """
        Parameters:
        - nin: number of inputs
        - layer_configs: list of tuples (nout, activation), e.g. [(4, 'relu'), (4, 'relu'), (1, 'linear')]
        """
        sizes = [nin] + [nout for nout, _ in layer_configs]
        activs = [act for _, act in layer_configs]
        self.layers = [Layer(sizes[i], sizes[i+1], activs[i]) for i in range(len(layer_configs))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"MLP(layers={[len(layer.neurons) for layer in self.layers]})"
