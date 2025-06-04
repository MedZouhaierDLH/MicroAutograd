from microautograd.engine import Value
import math

def mse_loss(pred: Value, target: Value) -> Value:
    """
    Mean Squared Error loss: (pred - target)^2
    
    This is the most straightforward loss function - it simply measures
    the squared difference between prediction and target.
    """
    return (pred - target) ** 2

def binary_cross_entropy_loss(pred: Value, target: Value, eps=1e-12) -> Value:
    """
    Binary Cross-Entropy loss: -[y*log(p) + (1-y)*log(1-p)]
    
    This loss function is designed for binary classification problems.
    The key insight here is that we must preserve the computational graph
    while still implementing numerical stability through clamping.
    
    Parameters:
    - pred: The model's prediction (should be output of sigmoid for proper interpretation)
    - target: The true binary label (0 or 1)
    - eps: Small epsilon value to prevent log(0) which would cause numerical issues
    
    The critical fix: Instead of creating a new Value object that breaks the
    computational graph, we use the new clamp() method that maintains gradient flow.
    """
    p = pred
    y = target
    
    # FIXED: Use clamp method instead of creating new Value object
    # This preserves the computational graph while preventing numerical issues
    p_clamped = p.clamp(eps, 1 - eps)
    
    # Now we can safely compute the cross-entropy loss
    # The (1 - p_clamped) computation also maintains the graph
    return -(y * p_clamped.log() + (1 - y) * (1 - p_clamped).log())

def categorical_cross_entropy_loss(predictions, targets):
    """
    Categorical Cross-Entropy loss for multi-class classification.
    
    This is included as an example of how you might extend the loss functions
    for more complex scenarios.
    """
    # This would require implementing softmax and handling multiple outputs
    # Left as an exercise for future extension
    raise NotImplementedError("Categorical cross-entropy not yet implemented")