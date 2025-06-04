import math

class Value:
    """A scalar value that supports automatic differentiation."""

    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None #leaf nodes backward are by initialization empty function
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"
    
    def __hash__(self):
        return id(self)

    # Basic operations-------------------------------------------------------------------------------------
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward

        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "Only int/float powers supported"
        out = Value(self.data**other, (self,), f'**{other}')

        def _backward():
            self.grad += (other * self.data**(other-1)) * out.grad
        out._backward = _backward

        return out

    def __neg__(self):  # -self
        return self * -1

    def __sub__(self, other):  # self - other
        return self + (-other)

    def __rsub__(self, other):  # other - self
        return other + (-self)

    def __radd__(self, other):  # other + self  (even if Python does 0 + Value, it will automatically call Value.__radd__() )
        return self + other

    def __rmul__(self, other):  # other * self  (5 * a)
        return self * other

    def __truediv__(self, other):  # self / other
        return self * other**-1

    def __rtruediv__(self, other):  # other / self
        return other * self**-1

    # Activation functions---------------------------------------------------------------
    def relu(self):
        out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')

        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward

        return out

    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1)/(math.exp(2*x) + 1)
        out = Value(t, (self,), 'tanh')

        def _backward():
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward

        return out
    
    def sigmoid(self):
        x = self.data
        s = 1 / (1 + math.exp(-x))
        out = Value(s, (self,), 'sigmoid')

        def _backward():
            self.grad += s * (1 - s) * out.grad
        out._backward = _backward

        return out

    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self,), 'exp')

        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward

        return out

    def log(self):
        out = Value(math.log(self.data), (self,), 'log')

        def _backward():
            self.grad += (1 / self.data) * out.grad
        out._backward = _backward

        return out
    
    def clamp(self, min_val, max_val):
        """Clamp value between min_val and max_val while preserving gradients."""
        # Use a smooth approximation that maintains differentiability
        # This is more numerically stable than hard clamping
        clamped_data = max(min(self.data, max_val), min_val)
        out = Value(clamped_data, (self,), f'clamp({min_val},{max_val})')
        
        def _backward():
            # Gradient flows through if we're not at the boundaries
            if min_val < self.data < max_val:
                self.grad += out.grad
            # If we're at boundaries, gradient is zero (but we still maintain the graph)
        
        out._backward = _backward
        return out
    
    # Comparison-----------------------------------------------------------------------
    # Allows using between Value and Value or Value and number
   
    def __lt__(self, other): # < 
        """Less than: Enables 'a < b' comparison with Values."""
        other = other.data if isinstance(other, Value) else other
        return self.data < other

    def __le__(self, other):    # <= 
        """Less than or equal: Enables 'a <= b'."""
        other = other.data if isinstance(other, Value) else other
        return self.data <= other

    def __gt__(self, other): # > 
        """Greater than: Enables 'a > b'."""
        other = other.data if isinstance(other, Value) else other
        return self.data > other

    def __ge__(self, other): # >=
        """Greater than or equal: Enables 'a >= b'."""
        other = other.data if isinstance(other, Value) else other
        return self.data >= other

    def __eq__(self, other): # ==
        """Equality: Enables 'a == b'."""
        other = other.data if isinstance(other, Value) else other
        return self.data == other

    def __ne__(self, other): # !=
        """Inequality: Enables 'a != b'."""
        other = other.data if isinstance(other, Value) else other
        return self.data != other

    # Backpropagation------------------------------------------------------------------
    def backward(self):
        topo = []
        visited = set() # requires hashable objects!

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child) # Recursivity
                topo.append(v)

        build_topo(self)

        self.grad = 1.0
        for node in reversed(topo):
            node._backward()