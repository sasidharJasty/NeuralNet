# Neural Network from Scratch - Iris & Titanic Classification

A complete implementation of neural networks from scratch using Python and NumPy, featuring both standard gradient descent and Adam optimization. This project includes classification examples on the Iris dataset and Titanic survival prediction.

## 📋 Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Neural Network Architecture](#neural-network-architecture)
- [Datasets](#datasets)
- [Results](#results)
- [Optimization Techniques](#optimization-techniques)
- [Visualization](#visualization)
- [Model Persistence](#model-persistence)
- [Contributing](#contributing)

## ✨ Features

- **Neural Network from Scratch**: Complete implementation without external ML libraries
- **Multiple Optimizers**: Standard SGD and Adam optimization
- **Model Persistence**: Save and load trained models
- **Feature Engineering**: Advanced preprocessing for Titanic dataset
- **Visualization**: Decision boundaries, feature importance, and performance metrics
- **Early Stopping**: Prevent overfitting during training
- **Threshold Optimization**: Automatic threshold tuning for binary classification

## 📁 Project Structure

```
ScratchNeuralNet/
├── main.py                 # Iris classification (3-class)
├── main2.py               # Titanic survival prediction (binary)
├── dense.py               # Standard dense layer implementation
├── dense_adam.py          # Adam optimizer dense layer
├── activations.py         # Activation functions (Tanh)
├── layer.py               # Base layer class
├── losses.py              # Loss functions (MSE)
├── network.py             # Training and prediction functions
├── outside.py             # Model saving/loading utilities
├── iris.csv               # Iris dataset
├── train.csv              # Titanic training data
├── test.csv               # Titanic test data
├── gender_submission.csv  # Titanic ground truth for testing
└── README.md              # This file
```

## 🛠 Installation

1. **Clone the repository:**
```bash
git clone <repository-url>
cd ScratchNeuralNet
```

2. **Install required packages:**
```bash
pip install numpy pandas matplotlib scikit-learn
```

3. **Download datasets:**
   - Iris dataset: Usually included or download from UCI ML Repository
   - Titanic dataset: Download from Kaggle Titanic competition

## 🚀 Usage

### Iris Classification (3-class)

```bash
python main.py
```

**Features:**
- 4 input features (sepal/petal length and width)
- 3-class classification (setosa, versicolor, virginica)
- Network architecture: 4 → 8 → 6 → 4 → 3
- Visualization of decision boundaries

### Titanic Survival Prediction (Binary)

```bash
python main2.py
```

**Features:**
- Enhanced feature engineering (9 engineered features)
- Binary classification (survived/died)
- Network architecture: 9 → 64 → 32 → 16 → 8 → 1
- Performance evaluation against test set

## 🧠 Neural Network Architecture

### Core Components

1. **Dense Layer** (`dense.py`):
```python
class Dense(Layer):
    def __init__(self, input_size, output_size)
    def forward(self, input)
    def backward(self, output_gradient, learning_rate)
```

2. **Adam Optimizer Layer** (`dense_adam.py`):
```python
class DenseAdam(Layer):
    def __init__(self, input_size, output_size, alpha=0.001, beta1=0.9, beta2=0.999)
    # Implements momentum and adaptive learning rates
```

3. **Activation Functions** (`activations.py`):
   - Tanh activation with derivative

### Network Configurations

**Iris Network:**
```python
network = [
    DenseAdam(4, 8, alpha=0.01),
    Tanh(),
    DenseAdam(8, 6, alpha=0.001),
    Tanh(),
    DenseAdam(6, 4, alpha=0.001),
    Tanh(),
    DenseAdam(4, 3, alpha=0.01),
    Tanh()
]
```

**Titanic Network:**
```python
network = [
    DenseAdam(9, 64, alpha=0.001),
    Tanh(),
    DenseAdam(64, 32, alpha=0.001),
    Tanh(),
    DenseAdam(32, 16, alpha=0.001),
    Tanh(),
    DenseAdam(16, 8, alpha=0.001),
    Tanh(),
    DenseAdam(8, 1, alpha=0.001),
    Tanh()
]
```

## 📊 Datasets

### Iris Dataset
- **Samples**: 150
- **Features**: 4 (sepal length/width, petal length/width)
- **Classes**: 3 (setosa, versicolor, virginica)
- **Task**: Multi-class classification

### Titanic Dataset
- **Training samples**: 891
- **Test samples**: 418
- **Original features**: 12
- **Engineered features**: 9
- **Task**: Binary survival prediction

#### Feature Engineering for Titanic:
```python
Features created:
- Title extraction from names (Mr, Mrs, Miss, Master, Rare)
- Family size (SibSp + Parch + 1)
- IsAlone flag
- Fare per person
- Age bands (5 categories)
- Fare bands (4 quartiles)
- Enhanced categorical encoding
```

## 📈 Results

### Iris Classification
- **Training Accuracy**: ~95-98%
- **Features**: All 3 classes correctly identified
- **Visualization**: 2D decision boundaries using first two features

### Titanic Prediction
- **Training Accuracy**: ~85-90%
- **Test Accuracy**: ~75-80%
- **Precision**: 0.742
- **Recall**: 0.645
- **F1-Score**: 0.690

## ⚡ Optimization Techniques

### Adam Optimizer Implementation
```python
# First moment (momentum)
m_t = β₁ * m_{t-1} + (1 - β₁) * ∇L/∇w_t

# Second moment (adaptive learning rate)
v_t = β₂ * v_{t-1} + (1 - β₂) * (∇L/∇w_t)²

# Bias correction
m̂_t = m_t / (1 - β₁^t)
v̂_t = v_t / (1 - β₂^t)

# Weight update
w_{t+1} = w_t - α * m̂_t / (√v̂_t + ε)
```

**Parameters:**
- α (learning rate): 0.001
- β₁ (momentum decay): 0.9
- β₂ (RMSprop decay): 0.999
- ε (numerical stability): 1e-8

### Training Optimizations
- **Early stopping**: Prevent overfitting
- **Learning rate scheduling**: Adaptive rates per layer
- **Threshold optimization**: Find optimal classification threshold
- **Feature normalization**: Z-score standardization

## 📊 Visualization

### Decision Boundaries
- 2D projections of multi-dimensional decision spaces
- Color-coded class predictions
- Actual data points overlaid

### Performance Metrics
- Confusion matrices
- Feature importance (correlation analysis)
- Training/validation curves
- Class distribution analysis

### Example Plots Generated:
1. **Decision Boundary Plot**: Age vs Fare for Titanic
2. **Feature Importance**: Correlation with survival
3. **Survival Distribution**: Pie charts
4. **3D Decision Surface**: Multi-dimensional visualization

## 💾 Model Persistence

### Saving Models
```python
save_model(network, "model.pkl", x_mean, x_std)
```

### Loading Models
```python
network, x_mean, x_std = load_model("model.pkl")
```

**Features:**
- Saves network weights and biases
- Preserves normalization parameters
- Handles both Dense and DenseAdam layers
- Automatic model detection and loading

## 🔧 Configuration

### Key Hyperparameters
```python
# Training
epochs = 1000-2000
learning_rate = 0.001-0.01
batch_size = 1 (sample-by-sample)

# Adam Optimizer
alpha = 0.001
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-8

# Classification
threshold = 0.45-0.5 (optimized)
```

## 📝 Usage Examples

### Training a New Model
```python
# For Iris
python main.py  # Trains and visualizes

# For Titanic
python main2.py  # Trains, tests, and evaluates
```

### Using Saved Models
```python
# Models are automatically loaded if they exist
# Delete .pkl files to force retraining
force_retrain()  # Utility function
```

### Custom Predictions
```python
# Normalize input
test_input = normalize_input(np.array([5.1, 3.5, 1.4, 0.2]))

# Predict
prediction = predict(network, test_input)
predicted_class = np.argmax(prediction)
```

## 🚀 Performance Tips

1. **Faster Training**: Reduce epochs for development
2. **Better Accuracy**: Increase network depth/width
3. **Visualization Speed**: Lower resolution for faster plotting
4. **Memory Usage**: Process data in batches for large datasets

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is open source and available under the MIT License.

## 🔍 Future Improvements

- [ ] Implement different activation functions (ReLU, Sigmoid)
- [ ] Add support for different loss functions (Cross-entropy)
- [ ] Implement batch processing
- [ ] Add regularization techniques (L1/L2, Dropout)
- [ ] GPU acceleration support
- [ ] Hyperparameter optimization
- [ ] Cross-validation support

## 📞 Contact

For questions or suggestions, please open an issue on GitHub.

---

**Note**: This implementation is for educational purposes to understand the fundamentals of neural networks. For production use, consider established frameworks like TensorFlow or PyTorch.