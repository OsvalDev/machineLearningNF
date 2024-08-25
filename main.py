import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Hypothesis Function
def hypothesisFunction(params, xFeatures):
    return np.dot(xFeatures, params)

# Mean Square Error Function
def meanSquareErrorFunction(params, xFeatures, yResults):
    m = len(yResults)
    yHypothesis = np.dot(xFeatures, params)
    error = yHypothesis - yResults
    return (1 / (2 * m)) * np.dot(error.T, error)

# Gradient Descent Function
def gradientDescentFunction(params, xFeatures, yResults, alpha):
    m = len(yResults)
    yHypothesis = np.dot(xFeatures, params)
    error = yHypothesis - yResults
    gradient = (1 / m) * np.dot(xFeatures.T, error)
    return params - alpha * gradient

# Scaling Function
def scalingFunction(xFeatures):
    xFeatures = np.array(xFeatures)
    for i in range(1, xFeatures.shape[1]):  # Skip bias column
        xFeatures[:, i] = (xFeatures[:, i] - np.mean(xFeatures[:, i])) / np.max(xFeatures[:, i])
    return xFeatures

# Process Data Set
# Load and clean data
dataset = pd.read_csv('./data/SeoulBikeData.csv')
dataset = dataset[['Hour', 'Temperature', 'Seasons', 'Rented Bike Count']]

# Convert categorical values to numeric
dataset['Seasons'] = dataset['Seasons'].map({'Spring': 0, 'Summer': 2, 'Autumn': 3, 'Winter': 4})

# Separate features (X) and target variable (y)
X = dataset[['Hour', 'Temperature', 'Seasons']].values
y = dataset['Rented Bike Count'].values

# Add bias column
X = np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)

# Split data into training, validation, and test sets
XTrain, XTemp, yTrain, yTemp = train_test_split(X, y, test_size=0.4, random_state=42)
XVal, XTest, yVal, yTest = train_test_split(XTemp, yTemp, test_size=0.5, random_state=42)

# Scale data
XTrainScaled = scalingFunction(XTrain)
XValScaled = scalingFunction(XVal)
XTestScaled = scalingFunction(XTest)

# Computation
params = np.zeros(XTrainScaled.shape[1])  # Initialize parameters
alpha = 0.7
epochs = 10000

trainErrors = []
valErrors = []

for i in range(epochs):
    params = gradientDescentFunction(params, XTrainScaled, yTrain, alpha)
    trainError = meanSquareErrorFunction(params, XTrainScaled, yTrain)
    valError = meanSquareErrorFunction(params, XValScaled, yVal)
    trainErrors.append(trainError)
    valErrors.append(valError)
    if i % 500 == 0 or i+1 == epochs :
        print(f"{i}: \t Training Error = {trainError} \t Validation Error = {valError}")

# Print Final Parameters
print(f"Final Parameters:{params}")
print("You can find graphs with result in results folder.")

# Make Predictions
yTrainPred = hypothesisFunction(params, XTrainScaled)
yValPred = hypothesisFunction(params, XValScaled)
yTestPred = hypothesisFunction(params, XTestScaled)

# Calculate R squared Value
def rSquared(yReal, yPred):
    ssTotal = np.sum((yReal - np.mean(yReal)) ** 2)
    ssResidual = np.sum((yReal - yPred) ** 2)
    return 1 - (ssResidual / ssTotal)

trainRSquared = rSquared(yTrain, yTrainPred)
valRSquared = rSquared(yVal, yValPred)

print(f"R squared for Training Set: {trainRSquared}")
print(f"R squared for Validation Set: {valRSquared}")

# Save Graphs
if not os.path.exists('results'):
    os.makedirs('results')

# Plot training and validation error
sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))
plt.plot(range(epochs), trainErrors, label='Training Error')
plt.plot(range(epochs), valErrors, label='Validation Error')
plt.xlabel('Epoch')
plt.ylabel('Mean Square Error')
plt.title('Training and Validation Error over Epochs')
plt.legend()
plt.grid(True)
plt.savefig('results/errorPlot.png')
plt.close()

# Plot comparison of expected vs predicted values
plt.figure(figsize=(14, 6))

# Training set
plt.subplot(1, 2, 1)
plt.plot(range(len(yTrain)), yTrain, label='Actual Train Values', color='blue')
plt.plot(range(len(yTrain)), yTrainPred, label='Predicted Train Values', color='red')
plt.xlabel('Index')
plt.ylabel('Rented Bike Count')
plt.title('Training Set: Actual vs Predicted')
plt.legend()
plt.grid(True)

# Test set
plt.subplot(1, 2, 2)
plt.plot(range(len(yTest)), yTest, label='Actual Test Values', color='blue')
plt.plot(range(len(yTest)), yTestPred, label='Predicted Test Values', color='red')
plt.xlabel('Index')
plt.ylabel('Rented Bike Count')
plt.title('Test Set: Actual vs Predicted')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('results/comparisonPlot.png')
plt.close()
