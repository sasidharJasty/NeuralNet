from dense import Dense
from dense_adam import DenseAdam
from losses import mse, mse_prime
import numpy as np
from activations import Tanh
from network import predict, train, train_with_early_stopping
import matplotlib.pyplot as plt
import pandas as pd

from outside import save_model, load_model, force_retrain
import os

# Model file path
model_file = "iris_neural_network_adam.pkl"

iris = pd.read_csv("iris.csv")
iris = iris.sample(frac=1).reset_index(drop=True)  # shuffle
# one-hot encoding
iris["species"] = iris["species"].astype("category").cat.codes
data = iris.values

# Normalize the input features for better training
x = data[:, 0:4]
x_mean = x.mean(axis=0)
x_std = x.std(axis=0)
x = (x - x_mean) / x_std  # Standardize features
x = x.reshape(-1, 4, 1)  # Shape: (samples, features, 1)

# One-hot encode the output for better classification
y_encoded = np.zeros((len(data), 3))
y_encoded[np.arange(len(data)), data[:, 4].astype(int)] = 1
y = y_encoded.reshape(-1, 3, 1)  # Shape: (samples, 3, 1)

# Check if model already exists
if os.path.exists(model_file):
    print("Loading existing model...")
    network, loaded_x_mean, loaded_x_std = load_model(model_file)
    x_mean, x_std = loaded_x_mean, loaded_x_std
    
    # Re-normalize data with loaded parameters
    x = data[:, 0:4]
    x = (x - x_mean) / x_std
    x = x.reshape(-1, 4, 1)
else:
    print("Training new model with Adam optimizer...")
    # Network using Adam optimizer
    network = [
        DenseAdam(4, 8, alpha=0.001),    # Adam layer with learning rate 0.001
        Tanh(),
        DenseAdam(8, 6, alpha=0.001),    # Adam layer
        Tanh(),
        DenseAdam(6, 4, alpha=0.001),    # Adam layer
        Tanh(),
        DenseAdam(4, 3, alpha=0.001),    # Adam output layer
        Tanh()
    ]
    
    # Train with fewer epochs since Adam converges faster
    train_with_early_stopping(network, mse, mse_prime, x, y, epochs=5000, learning_rate=0.001)
    
    # Save the trained model
    save_model(network, model_file, x_mean, x_std)

# Test accuracy
correct = 0
for i in range(len(x)):
    prediction = predict(network, x[i])
    predicted_class = np.argmax(prediction)
    actual_class = np.argmax(y[i])
    if predicted_class == actual_class:
        correct += 1

accuracy = correct / len(x)
print(f"Training Accuracy with Adam: {accuracy:.2%}")

# decision boundary plot using first two features
points = []
x_min, x_max = x[:, 0, 0].min() - 0.5, x[:, 0, 0].max() + 0.5
y_min, y_max = x[:, 1, 0].min() - 0.5, x[:, 1, 0].max() + 0.5

for x_val in np.linspace(x_min, x_max, 30):  # Increased resolution
    for y_val in np.linspace(y_min, y_max, 30):
        # Use mean values for the other two features (already normalized)
        test_input = np.array([[x_val], [y_val], [x[:, 2, 0].mean()], [x[:, 3, 0].mean()]])
        z = predict(network, test_input)
        predicted_class = np.argmax(z)
        points.append([x_val, y_val, predicted_class])

points = np.array(points)

fig = plt.figure(figsize=(12, 5))

# Plot decision boundary
ax1 = fig.add_subplot(121)
scatter = ax1.scatter(points[:, 0], points[:, 1], c=points[:, 2], alpha=0.3, cmap="viridis")
ax1.scatter(x[:, 0, 0], x[:, 1, 0], c=np.argmax(y.reshape(-1, 3), axis=1), cmap="viridis", edgecolors='black')
ax1.set_xlabel('Feature 1 (Sepal Length)')
ax1.set_ylabel('Feature 2 (Sepal Width)')
ax1.set_title('Decision Boundary')
plt.colorbar(scatter, ax=ax1)

# 3D visualization
ax2 = fig.add_subplot(122, projection="3d")
ax2.scatter(points[:, 0], points[:, 1], points[:, 2], c=points[:, 2], cmap="viridis", alpha=0.6)
ax2.set_xlabel('Feature 1')
ax2.set_ylabel('Feature 2')
ax2.set_zlabel('Predicted Class')
ax2.set_title('3D Decision Surface')

plt.tight_layout()
plt.show()

# Helper function to normalize new inputs
def normalize_input(input_data):
    normalized = (input_data - x_mean) / x_std
    return normalized.reshape(-1, 1)

# Test with properly normalized inputs
test1 = normalize_input(np.array([5.1, 3.5, 1.4, 0.2]))  # Expected class 0 (setosa)
test2 = normalize_input(np.array([6.0, 2.2, 4.0, 1.0]))  # Expected class 1 (versicolor)
test3 = normalize_input(np.array([7.0, 3.0, 6.0, 2.0]))  # Expected class 2 (virginica)

print("Test predictions:")
pred1 = predict(network, test1)
pred2 = predict(network, test2)
pred3 = predict(network, test3)

print(f"Test 1 - Raw output: {pred1.flatten()}, Predicted class: {np.argmax(pred1)} (Expected: 0)")
print(f"Test 2 - Raw output: {pred2.flatten()}, Predicted class: {np.argmax(pred2)} (Expected: 1)")
print(f"Test 3 - Raw output: {pred3.flatten()}, Predicted class: {np.argmax(pred3)} (Expected: 2)")

# Also check the class distribution in your training data
print(f"\nClass distribution in training data:")
classes = np.argmax(y.reshape(-1, 3), axis=1)
for i in range(3):
    count = np.sum(classes == i)
    print(f"Class {i}: {count} samples")

# Optional: Force retrain by deleting the model file


# Uncomment the line below to force retraining next time
# force_retrain()
