

def predict(network, input):
    output = input
    for layer in network:
        output = layer.forward(output)
    return output

def train(network, loss, loss_prime, x_train, y_train, epochs = 1000, learning_rate = 0.01, verbose = True):
    for e in range(epochs):
        error = 0
        for x, y in zip(x_train, y_train):
            # forward
            output = predict(network, x)

            # error
            error += loss(y, output)

            # backward
            grad = loss_prime(y, output)
            for layer in reversed(network):
                grad = layer.backward(grad, learning_rate)

        error /= len(x_train)
        if verbose:
            print(f"{e + 1}/{epochs}, error={error}")

import numpy as np

def predict_batch(network, inputs):
    """Predict multiple inputs at once for faster visualization"""
    outputs = []
    for input_sample in inputs:
        output = input_sample
        for layer in network:
            output = layer.forward(output)
        outputs.append(output)
    return np.array(outputs)

def train_with_early_stopping(network, loss, loss_prime, x_train, y_train, epochs=1000, learning_rate=0.01, patience=50):
    """Training with early stopping to prevent overtraining"""
    best_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        total_loss = 0
        for x_sample, y_sample in zip(x_train, y_train):
            # Forward pass
            output = x_sample
            for layer in network:
                output = layer.forward(output)
            
            # Calculate loss
            total_loss += loss(y_sample, output)
            
            # Backward pass
            grad = loss_prime(y_sample, output)
            for layer in reversed(network):
                grad = layer.backward(grad, learning_rate)
        
        avg_loss = total_loss / len(x_train)
        
        # Early stopping check
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
            
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {avg_loss:.4f}")
    
    return network