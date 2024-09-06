import numpy as np
import pandas as pd
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
    for i in range(1, xFeatures.shape[1]):
        xFeatures[:, i] = (xFeatures[:, i] - np.mean(xFeatures[:, i])) / np.max(xFeatures[:, i])
    return xFeatures

import numpy as np

# Function to split de datset in train and test
def trainTestSplit(X, y, testSize=0.8, randomState=None):
    if randomState is not None:
        np.random.seed(randomState)
    
    # Shuffle indexes
    indexes = np.arange(X.shape[0])
    np.random.shuffle(indexes)
    
    # Calculate split index
    split_index = int(len(indexes) * testSize)
    
    # Split the indexes
    testIndexes = indexes[:split_index]
    trainIndexes = indexes[split_index:]
    
    # Split the data
    XTrain, XTest = X[trainIndexes], X[testIndexes]
    yTrain, yTest = y[trainIndexes], y[testIndexes]
    
    return XTrain, XTest, yTrain, yTest

# Calculate R squared Value
def rSquared(yReal, yPred):
    ssTotal = np.sum((yReal - np.mean(yReal)) ** 2)
    ssResidual = np.sum((yReal - yPred) ** 2)
    return 1 - (ssResidual / ssTotal)

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
XTrain, XTemp, yTrain, yTemp = trainTestSplit(X, y, testSize=0.4, randomState=42)
XVal, XTest, yVal, yTest = trainTestSplit(XTemp, yTemp, testSize=0.5, randomState=42)

# Scale data
XTrainScaled = scalingFunction(XTrain)
XValScaled = scalingFunction(XVal)
XTestScaled = scalingFunction(XTest)

# Computation
params = np.zeros(XTrainScaled.shape[1])  # Initialize parameters
alpha = 0.01
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
plt.figure(figsize=(12, 6))

# Gráfico de dispersión para el conjunto de entrenamiento
plt.subplot(1, 2, 1)
plt.scatter(yTrain, yTrainPred, color='blue', alpha=0.5)
plt.plot([min(yTrain), max(yTrain)], [min(yTrain), max(yTrain)], color='red', linewidth=2)  # Línea y=x
plt.xlabel('Actual Values (Train)')
plt.ylabel('Predicted Values (Train)')
plt.title('Training Set: Actual vs Predicted')
plt.grid(True)

# Gráfico de dispersión para el conjunto de prueba
plt.subplot(1, 2, 2)
plt.scatter(yTest, yTestPred, color='green', alpha=0.5)
plt.plot([min(yTest), max(yTest)], [min(yTest), max(yTest)], color='red', linewidth=2)  # Línea y=x
plt.xlabel('Actual Values (Test)')
plt.ylabel('Predicted Values (Test)')
plt.title('Test Set: Actual vs Predicted')
plt.grid(True)


plt.tight_layout()
plt.savefig('results/comparisonPlot.png')
plt.close()
