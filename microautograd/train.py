from microautograd.engine import Value
from microautograd.losses import mse_loss, binary_cross_entropy_loss
from microautograd.nn import MLP
import random

def zero_grad(params):
    """
    Reset gradients of all model parameters.
    
    This is crucial before each backward pass because gradients accumulate
    by default. If we don't zero them out, we'd be adding gradients from
    the current batch to gradients from previous batches, which would
    completely mess up our training process.
    """
    for p in params:
        p.grad = 0.0

def train(model, X_raw, Y_raw, loss_fn, epochs=100, lr=0.01, verbose=True):
    """
    Train a model on a given dataset using a given loss function.

    This training function implements the standard machine learning training loop:
    1. Forward pass: compute predictions and loss
    2. Backward pass: compute gradients via backpropagation  
    3. Update step: adjust parameters using gradient descent
    4. Repeat for specified number of epochs

    Parameters:
    - model: MLP instance to train
    - X_raw: list of input lists (floats) - our training features
    - Y_raw: list of target float values or Value instances - our training labels
    - loss_fn: function(Value, Value) -> Value - determines what we're optimizing
    - epochs: number of complete passes through the dataset
    - lr: learning rate - controls how big steps we take during optimization
    - verbose: if True, prints loss periodically to monitor training progress

    Returns:
    - model: the trained model (parameters updated in-place)
    """
    # Convert targets to Value objects if they aren't already
    # This ensures consistency in our computational graph
    Y = [y if isinstance(y, Value) else Value(y) for y in Y_raw]

    print(f"Starting training for {epochs} epochs...")
    print(f"Dataset size: {len(X_raw)} samples")
    print(f"Learning rate: {lr}")
    print(f"Loss function: {loss_fn.__name__}")
    print("-" * 50)

    for epoch in range(epochs):
        # Initialize total loss for this epoch
        total_loss = Value(0.0)

        # Forward pass: compute predictions and accumulate loss
        # We process the entire dataset in each epoch
        for x, y_true in zip(X_raw, Y):
            # Convert input features to Value objects for automatic differentiation
            x_val = [Value(xi) for xi in x]
            
            # Get model prediction
            y_pred = model(x_val)
            
            # Compute loss for this sample
            loss = loss_fn(y_pred, y_true)
            
            # Accumulate loss across all samples
            total_loss += loss

        # Backward pass: compute gradients
        # First, we must zero out gradients from the previous iteration
        zero_grad(model.parameters())
        
        # Then we compute gradients by calling backward on our total loss
        # This triggers the chain rule through the entire computational graph
        total_loss.backward()

        # Update step: gradient descent
        # We update each parameter by moving in the opposite direction of its gradient
        # The learning rate controls how big a step we take
        for p in model.parameters():
            p.data -= lr * p.grad

        # Logging: print progress periodically
        if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
            avg_loss = total_loss.data / len(X_raw)  # Average loss per sample
            print(f"Epoch {epoch:03d} | Total Loss = {total_loss.data:.6f} | Avg Loss = {avg_loss:.6f}")

    print("\nTraining completed!")
    return model

def evaluate_model(model, X_test, Y_test, loss_fn):
    """
    Evaluate model performance on test data without updating parameters.
    
    This function is useful for checking how well your model generalizes
    to unseen data. It computes the same loss as during training, but
    crucially, it doesn't call backward() or update any parameters.
    """
    total_loss = Value(0.0)
    predictions = []
    
    for x, y_true in zip(X_test, Y_test):
        x_val = [Value(xi) for xi in x]
        y_pred = model(x_val)
        
        loss = loss_fn(y_pred, y_true if isinstance(y_true, Value) else Value(y_true))
        total_loss += loss
        predictions.append(y_pred.data)
    
    avg_loss = total_loss.data / len(X_test)
    return avg_loss, predictions

# ========== Example Usage ==========
if __name__ == "__main__":
    print("=== Linear Regression Example ===")
    # Sample dataset: learn y = 2x1 + x2 - 3
    X = [
        [2.0, 3.0],   # Expected output: 2*2 + 3 - 3 = 4
        [1.0, -1.0],  # Expected output: 2*1 + (-1) - 3 = -2  
        [0.5, 2.0],   # Expected output: 2*0.5 + 2 - 3 = 0
        [-1.0, -2.0], # Expected output: 2*(-1) + (-2) - 3 = -7
    ]
    Y = [2*x1 + x2 - 3 for x1, x2 in X]  # True function we want to learn
    
    print("True function: y = 2*x1 + x2 - 3")
    print("Target outputs:", Y)

    # Build model: 2 inputs -> 4 ReLU -> 4 ReLU -> 1 linear output
    model = MLP(2, [(4, "relu"), (4, "relu"), (1, "linear")])
    
    # Train model using mean squared error loss
    print(f"\nInitial model parameters: {len(model.parameters())} total")
    trained_model = train(model, X, Y, loss_fn=mse_loss, epochs=200, lr=0.01)
    
    # Test the trained model
    print("\n=== Testing Trained Model ===")
    test_loss, predictions = evaluate_model(trained_model, X, Y, mse_loss)
    print(f"Final test loss: {test_loss:.6f}")
    
    print("\nPredictions vs Targets:")
    for i, (pred, target) in enumerate(zip(predictions, Y)):
        print(f"Sample {i}: Predicted = {pred:.4f}, Target = {target:.4f}, Error = {abs(pred - target):.4f}")
    
    print("\n=== Binary Classification Example ===")
    # Simple binary classification: predict if x1 + x2 > 0
    X_binary = [
        [1.0, 2.0],   # 1 + 2 = 3 > 0, so label = 1
        [-1.0, -1.0], # -1 + -1 = -2 < 0, so label = 0
        [2.0, -1.0],  # 2 + -1 = 1 > 0, so label = 1
        [-2.0, 1.0],  # -2 + 1 = -1 < 0, so label = 0
    ]
    Y_binary = [1.0 if x1 + x2 > 0 else 0.0 for x1, x2 in X_binary]
    
    print("Binary classification: predict if x1 + x2 > 0")
    print("Target outputs:", Y_binary)
    
    # Build model with sigmoid output for binary classification
    binary_model = MLP(2, [(4, "relu"), (1, "sigmoid")])
    
    # Train using binary cross-entropy loss
    trained_binary_model = train(binary_model, X_binary, Y_binary, 
                                loss_fn=binary_cross_entropy_loss, epochs=100, lr=0.1)
    
    # Test the binary model
    print("\n=== Testing Binary Classification Model ===")
    test_loss, binary_predictions = evaluate_model(trained_binary_model, X_binary, Y_binary, binary_cross_entropy_loss)
    print(f"Final test loss: {test_loss:.6f}")
    
    print("\nBinary Predictions vs Targets:")
    for i, (pred, target) in enumerate(zip(binary_predictions, Y_binary)):
        predicted_class = 1 if pred > 0.5 else 0
        print(f"Sample {i}: Predicted = {pred:.4f} (class {predicted_class}), Target = {int(target)}")