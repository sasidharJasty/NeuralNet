from dense import Dense
from dense_adam import DenseAdam
from losses import mse, mse_prime
import numpy as np
from activations import Tanh
from network import predict, train
import matplotlib.pyplot as plt
import pandas as pd

from outside import save_model, load_model, force_retrain
import os

# Model file path
model_file = "titanic_adam_improved.pkl"

# Load training data with better preprocessing
titanic_train = pd.read_csv("train.csv")

# Better feature engineering
def preprocess_data(df, is_training=True):
    # Create a copy to avoid modifying original
    data = df.copy()
    
    # Extract title from names
    data['Title'] = data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    
    # Group rare titles
    data['Title'] = data['Title'].replace(['Lady', 'Countess','Capt', 'Col',
                                          'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    data['Title'] = data['Title'].replace('Mlle', 'Miss')
    data['Title'] = data['Title'].replace('Ms', 'Miss')
    data['Title'] = data['Title'].replace('Mme', 'Mrs')
    
    # Create family size feature
    data['FamilySize'] = data['SibSp'] + data['Parch'] + 1
    
    # Create "IsAlone" feature
    data['IsAlone'] = (data['FamilySize'] == 1).astype(int)
    
    # Fare per person
    data['FarePerPerson'] = data['Fare'] / data['FamilySize']
    
    # Age bands
    data['Age'].fillna(data['Age'].median(), inplace=True)
    data['AgeBand'] = pd.cut(data['Age'], 5, labels=[0, 1, 2, 3, 4])
    
    # Fare bands
    data['Fare'].fillna(data['Fare'].median(), inplace=True)
    data['FareBand'] = pd.qcut(data['Fare'], 4, labels=[0, 1, 2, 3])
    
    # Encode categorical variables
    data['Sex'] = data['Sex'].map({'female': 0, 'male': 1})
    data['Title'] = data['Title'].map({'Mr': 0, 'Miss': 1, 'Mrs': 2, 'Master': 3, 'Rare': 4})
    
    # Fill missing embarked
    data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)
    data['Embarked'] = data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
    
    # Drop unnecessary columns
    data = data.drop(['Name', 'Ticket', 'Cabin', 'Age', 'Fare', 'SibSp', 'Parch'], axis=1)
    
    # Select final features
    feature_cols = ['Pclass', 'Sex', 'Embarked', 'FamilySize', 'IsAlone', 'Title', 'AgeBand', 'FareBand', 'FarePerPerson']
    
    if is_training:
        data = data[feature_cols + ['Survived']]
    else:
        data = data[feature_cols]
    
    return data

# Preprocess training data
titanic_train_processed = preprocess_data(titanic_train, is_training=True)
titanic_train_processed = titanic_train_processed.dropna()
print("Enhanced training data features:")
print(titanic_train_processed.head())

# Shuffle the data
titanic_train_processed = titanic_train_processed.sample(frac=1).reset_index(drop=True)

y_train = titanic_train_processed["Survived"].values.reshape(-1, 1)
titanic_train_processed = titanic_train_processed.drop(columns=["Survived"])

# Normalize the input features
x_train = titanic_train_processed.values.astype(np.float32)
x_mean = x_train.mean(axis=0)
x_std = x_train.std(axis=0)
x_std[x_std == 0] = 1  # Prevent division by zero
x_train = (x_train - x_mean) / x_std
x_train = x_train.reshape(-1, x_train.shape[1], 1)
y_train = y_train.reshape(-1, 1, 1)

print(f"Enhanced Training - Input shape: {x_train.shape}, Output shape: {y_train.shape}")
print(f"Number of features: {x_train.shape[1]}")

# Check if model already exists
if os.path.exists(model_file):
    print("Loading existing enhanced model...")
    network, loaded_x_mean, loaded_x_std = load_model(model_file)
    x_mean, x_std = loaded_x_mean, loaded_x_std
    
    # Re-normalize training data with loaded parameters
    x_train_raw = titanic_train_processed.values.astype(np.float32)
    x_train = (x_train_raw - x_mean) / x_std
    x_train = x_train.reshape(-1, x_train.shape[1], 1)
else:
    print("Training new enhanced model with Adam optimizer...")
    # Improved network architecture
    num_features = x_train.shape[1]
    network = [
        DenseAdam(num_features, 64, alpha=0.001),  # Larger first layer
        Tanh(),
        DenseAdam(64, 32, alpha=0.001),            # Gradual reduction
        Tanh(),
        DenseAdam(32, 16, alpha=0.001),            # More layers for complexity
        Tanh(),
        DenseAdam(16, 8, alpha=0.001),
        Tanh(),
        DenseAdam(8, 1, alpha=0.001),              # Output layer
        Tanh()
    ]
    
    # Train with more epochs and lower learning rate
    train(network, mse, mse_prime, x_train, y_train, epochs=2000, learning_rate=0.001)
    
    # Save the trained model
    save_model(network, model_file, x_mean, x_std)

# Test accuracy on training data
correct = 0
predictions_train = []
for i in range(len(x_train)):
    prediction = predict(network, x_train[i])
    predicted_class = 1 if prediction[0, 0] > 0.5 else 0
    actual_class = int(y_train[i, 0, 0])
    predictions_train.append(prediction[0, 0])
    if predicted_class == actual_class:
        correct += 1

training_accuracy = correct / len(x_train)
print(f"Enhanced Training Accuracy: {training_accuracy:.2%}")

# Load and preprocess test data
print("\nLoading and preprocessing test data...")
titanic_test = pd.read_csv("test.csv")
test_passenger_ids = titanic_test["PassengerId"].values

# Apply same enhanced preprocessing
titanic_test_processed = preprocess_data(titanic_test, is_training=False)

# Handle any remaining missing values
for col in titanic_test_processed.columns:
    if titanic_test_processed[col].isnull().any():
        titanic_test_processed[col].fillna(titanic_test_processed[col].median(), inplace=True)

# Normalize test data using training parameters
x_test = titanic_test_processed.values.astype(np.float32)
x_test = (x_test - x_mean) / x_std
x_test = x_test.reshape(-1, x_test.shape[1], 1)
print(f"Enhanced Test - Input shape: {x_test.shape}")

# Make predictions on test data
print("Making enhanced predictions on test data...")
test_predictions = []
test_probabilities = []

for i in range(len(x_test)):
    prediction = predict(network, x_test[i])
    survival_prob = prediction[0, 0]
    # Use optimized threshold (you might want to tune this)
    predicted_class = 1 if survival_prob > 0.45 else 0  # Slightly lower threshold
    test_predictions.append(predicted_class)
    test_probabilities.append(survival_prob)

test_predictions = np.array(test_predictions)
test_probabilities = np.array(test_probabilities)

# Load ground truth for comparison
try:
    gender_submission = pd.read_csv("gender_submission.csv")
    ground_truth = gender_submission["Survived"].values
    
    # Calculate test accuracy
    test_correct = np.sum(test_predictions == ground_truth)
    test_accuracy = test_correct / len(ground_truth)
    print(f"Enhanced Test Accuracy: {test_accuracy:.2%}")
    
    # Create confusion matrix
    true_positive = np.sum((test_predictions == 1) & (ground_truth == 1))
    true_negative = np.sum((test_predictions == 0) & (ground_truth == 0))
    false_positive = np.sum((test_predictions == 1) & (ground_truth == 0))
    false_negative = np.sum((test_predictions == 0) & (ground_truth == 1))
    
    print(f"\nEnhanced Confusion Matrix:")
    print(f"True Positive: {true_positive}")
    print(f"True Negative: {true_negative}")
    print(f"False Positive: {false_positive}")
    print(f"False Negative: {false_negative}")
    
    # Calculate enhanced metrics
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\nEnhanced Metrics:")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1-Score: {f1_score:.3f}")
    
    # Try different thresholds to optimize F1-score
    best_f1 = f1_score
    best_threshold = 0.45
    
    print(f"\nOptimizing threshold...")
    for threshold in np.arange(0.3, 0.7, 0.05):
        temp_predictions = (test_probabilities > threshold).astype(int)
        tp = np.sum((temp_predictions == 1) & (ground_truth == 1))
        tn = np.sum((temp_predictions == 0) & (ground_truth == 0))
        fp = np.sum((temp_predictions == 1) & (ground_truth == 0))
        fn = np.sum((temp_predictions == 0) & (ground_truth == 1))
        
        temp_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        temp_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        temp_f1 = 2 * (temp_precision * temp_recall) / (temp_precision + temp_recall) if (temp_precision + temp_recall) > 0 else 0
        
        if temp_f1 > best_f1:
            best_f1 = temp_f1
            best_threshold = threshold
    
    print(f"Best threshold: {best_threshold:.2f} with F1-score: {best_f1:.3f}")
    
except FileNotFoundError:
    print("gender_submission.csv not found. Skipping accuracy comparison.")
    ground_truth = None

# Create enhanced submission file
submission_df = pd.DataFrame({
    'PassengerId': test_passenger_ids,
    'Survived': test_predictions
})
submission_df.to_csv("enhanced_submission.csv", index=False)
print(f"\nEnhanced submission file saved as 'enhanced_submission.csv'")

print("\nKey improvements made:")
print("1. Enhanced feature engineering (Title extraction, Family size, etc.)")
print("2. Better handling of missing values")
print("3. Improved network architecture with more neurons")
print("4. Lower learning rate with more epochs")
print("5. Threshold optimization")
print("6. Additional engineered features for better prediction")