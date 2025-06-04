# MicroAutograd 🧠⚡

A minimal but powerful automatic differentiation engine implemented in pure Python. MicroAutograd demonstrates the core principles behind modern deep learning frameworks like PyTorch and TensorFlow, but in a codebase small enough to understand completely.

## 🎯 What is MicroAutograd?

MicroAutograd is a scalar-valued automatic differentiation (autograd) engine with a small neural network library built on top. It implements backpropagation over a dynamically built DAG (Directed Acyclic Graph) and supports:

- **Automatic differentiation** with reverse-mode backpropagation
- **Neural network components** (neurons, layers, MLPs)
- **Common activation functions** (ReLU, Tanh, Sigmoid)
- **Loss functions** (MSE, Binary Cross-Entropy)
- **Training utilities** with gradient descent optimization
- **Visualization tools** for computational graphs

## 🚀 Quick Start

### Basic Usage

```python
from engine import Value

# Create values and build a computational graph
x = Value(2.0, label='x')
y = Value(3.0, label='y')
z = x * y + x
z.label = 'z'

# Compute gradients via backpropagation
z.backward()

print(f"z = {z.data}")      # z = 8.0
print(f"dz/dx = {x.grad}")  # dz/dx = 4.0 (gradient of z with respect to x)
print(f"dz/dy = {y.grad}")  # dz/dy = 2.0 (gradient of z with respect to y)
```

### Neural Network Example

```python
from nn import MLP
from losses import mse_loss
from train import train

# Create a dataset: learn y = 2x1 + x2 - 3
X = [[2.0, 3.0], [1.0, -1.0], [0.5, 2.0], [-1.0, -2.0]]
Y = [2*x1 + x2 - 3 for x1, x2 in X]

# Build a neural network: 2 inputs -> 4 ReLU -> 4 ReLU -> 1 linear output
model = MLP(2, [(4, "relu"), (4, "relu"), (1, "linear")])

# Train the model
trained_model = train(model, X, Y, loss_fn=mse_loss, epochs=200, lr=0.01)
```

### Visualization

```python
from visualize import visualize_computation

# Build a computation
x = Value(2.0, label='x')
y = Value(-3.0, label='y')
z = x * y
z.label = 'output'

# Visualize the computational graph
z.backward()
visualize_computation(z, filename='computation_graph', show_gradients=True)
```

## 📁 Project Structure

```
MicroAutograd/
├── _pycache_/                  # Python bytecode cache (auto-generated)
├── microautograd/              # Package directory
│   ├── engine.py              # Core autodiff engine with Value class
│   ├── losses.py              # Loss functions (MSE, cross-entropy, etc.)
│   ├── nn.py                  # Neural network components (Neuron, Layer, MLP)
│   ├── train.py               # Training utilities and main training loop
│   └── visualize.py           # Visualization tools for computation graphs
├── micrograd_readme.md         # Documentation/README file
└── tutorial.ipynb             # Jupyter notebook tutorial
```

## 🔧 Core Components

### `engine.py` - The Heart of Autograd

The `Value` class is the core building block that wraps scalars and builds a computational graph:

```python
class Value:
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data        # The actual scalar value
        self.grad = 0.0        # Gradient (∂L/∂self)
        self._backward = lambda: None  # Function to compute gradients
        self._prev = set(_children)    # Parent nodes in the graph
        self._op = _op         # Operation that created this node
```

**Supported Operations:**
- Arithmetic: `+`, `-`, `*`, `/`, `**`
- Activation functions: `tanh()`, `relu()`, `sigmoid()`, `exp()`, `log()`
- Comparisons: `<`, `<=`, `>`, `>=`, `==`, `!=`
- Clamping: `clamp(min_val, max_val)` for numerical stability

### `nn.py` - Neural Network Building Blocks

**Neuron**: Single neuron with weights, bias, and activation function
```python
neuron = Neuron(nin=3, activation="relu")  # 3 inputs, ReLU activation
output = neuron([1.0, 2.0, 3.0])          # Forward pass
```

**Layer**: Collection of neurons
```python
layer = Layer(nin=3, nout=4, activation="tanh")  # 3->4 layer with tanh
```

**MLP**: Multi-layer perceptron
```python
model = MLP(2, [(4, "relu"), (4, "relu"), (1, "sigmoid")])  # 2->4->4->1 network
```

### `losses.py` - Loss Functions

**Mean Squared Error**: For regression tasks
```python
loss = mse_loss(prediction, target)  # (pred - target)²
```

**Binary Cross-Entropy**: For binary classification
```python
loss = binary_cross_entropy_loss(prediction, target)  # Handles numerical stability
```

### `train.py` - Training Loop

Complete training pipeline with gradient descent:
```python
def train(model, X, Y, loss_fn, epochs=100, lr=0.01):
    # Forward pass, backward pass, parameter update
    # Includes progress monitoring and evaluation utilities
```

### `visualize.py` - Graph Visualization

Create beautiful visualizations of computational graphs using Graphviz:
```python
visualize_computation(root_node, show_gradients=True, inline=True)  # For Jupyter
compare_before_after_backward(root_node)  # See gradient flow
```

## 🎓 Educational Features

### Mathematical Intuition

MicroAutograd is designed to make the mathematics of deep learning transparent:

- **Gradient Flow**: See exactly how gradients flow through operations
- **Chain Rule**: Watch the chain rule in action during backpropagation  
- **Activation Functions**: Understand how different activations affect gradient flow
- **Loss Landscapes**: Visualize how different loss functions behave

### Example Use Cases

1. **Linear Regression**: Learn linear relationships with MSE loss
2. **Binary Classification**: Classify data with sigmoid activation and cross-entropy loss
3. **Nonlinear Function Approximation**: Use MLPs to approximate complex functions
4. **Gradient Debugging**: Visualize computational graphs to debug gradient flow

## 🔍 Key Learning Concepts

### Automatic Differentiation
- **Forward Pass**: Build computational graph while computing values
- **Backward Pass**: Traverse graph in reverse, applying chain rule
- **Dynamic Graphs**: Graph is built during execution, not predefined

### Neural Network Training
- **Gradient Descent**: Iteratively adjust parameters opposite to gradient direction
- **Backpropagation**: Efficient algorithm for computing gradients in neural networks
- **Loss Functions**: Different objectives lead to different gradient behaviors

### Gradient Flow Insights
- **Addition**: Distributes gradients equally (`∂(a+b)/∂a = 1`)
- **Multiplication**: Scales gradients by other operand (`∂(a*b)/∂a = b`)  
- **Activations**: Introduce nonlinearity and affect gradient magnitude

## 🧪 Running Examples

```bash
# Basic automatic differentiation
python -c "
from engine import Value
x = Value(2.0); y = Value(3.0)
z = x * y + x; z.backward()
print(f'x.grad = {x.grad}, y.grad = {y.grad}')
"

# Neural network training
python train.py

# Create computational graph visualization (requires graphviz)
python -c "
from engine import Value
from visualize import visualize_computation
x = Value(2.0, label='x')
y = Value(-3.0, label='y') 
z = (x * y).tanh()
z.backward()
visualize_computation(z, filename='example')
"
```

## 📚 Dependencies

### Core functionality (no dependencies):
- Pure Python 3.6+
- Uses only standard library (`math`, `random`)

### Visualization (optional):
- `graphviz` - for computational graph visualization
  ```bash
  pip install graphviz
  ```

### Jupyter integration (optional):
- `IPython` - for inline visualization in notebooks
  ```bash
  pip install ipython jupyter
  ```

## 🎯 Learning Path

1. **Start with `engine.py`**: Understand how `Value` implements autograd
2. **Explore basic operations**: Try addition, multiplication, and activation functions
3. **Build simple neural networks**: Use `nn.py` to create MLPs
4. **Train your first model**: Use `train.py` examples  
5. **Visualize everything**: Use `visualize.py` to see the computational graph
6. **Experiment**: Try different architectures, loss functions, and datasets

## 🔬 Advanced Features

### Numerical Stability
- Gradient clamping to prevent exploding gradients
- Epsilon handling in logarithmic operations
- Smooth clamping function that preserves differentiability

### Debugging Support
- Rich `__repr__` methods for easy inspection
- Computational graph visualization
- Before/after backward pass comparisons
- Gradient flow tracing

### Extensibility
- Easy to add new operations (just implement forward and backward)
- Modular design allows swapping components
- Clear separation between engine, networks, and training

## 🤝 Contributing

This is an educational project! Contributions that improve clarity, add educational value, or fix bugs are welcome:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📖 Inspiration & References

This project was inspired by Andrej Karpathy's educational content:
- [The spelled-out intro to neural networks and backpropagation: building micrograd](https://www.youtube.com/watch?v=VMj-3S1tku0) - YouTube video
- [micrograd](https://github.com/karpathy/micrograd) - Original micrograd repository  
- [Automatic Differentiation in Machine Learning](https://arxiv.org/abs/1502.05767) - Academic foundation

## 🏆 What You'll Learn

By studying and experimenting with MicroAutograd, you'll gain deep understanding of:

- How automatic differentiation really works under the hood
- Why the chain rule is fundamental to deep learning  
- How gradients flow through different types of operations
- What neural networks are actually computing
- Why certain design choices (initialization, activation functions, loss functions) matter
- How to debug gradient flow problems
- The connection between calculus and machine learning

## 📄 License

MIT License - feel free to use this for learning, teaching, or building upon!

---

**Happy Learning! 🚀** 

*Remember: The best way to understand neural networks is to build one from scratch.*#   M i c r o A u t o g r a d 
 
 
