import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Process Data Set
# Load and clean data
dataSet = pd.read_csv('./data/SeoulBikeDataProcessed.csv')

# Separate features (X) and target variable (y)
x = dataSet.drop(columns=['rentedBikeCount'])
y = dataSet['rentedBikeCount'].values

# Split data into training, validation, and test sets
xTrain, xTemp, yTrain, yTemp = train_test_split(x, y, test_size=0.4, random_state=42)
xVal, xTest, yVal, yTest = train_test_split(xTemp, yTemp, test_size=0.5, random_state=42)

# Scale data
scaler = StandardScaler()
xTrainScaled = scaler.fit_transform(xTrain)
xValScaled = scaler.transform(xVal)
xTestScaled = scaler.transform(xTest)

# Initialize Linear Regression model
model = LinearRegression()

# Train the model
model.fit(xTrainScaled, yTrain)

# Make Predictions
yTrainPred = model.predict(xTrainScaled)
yValPred = model.predict(xValScaled)
yTestPred = model.predict(xTestScaled)

# Calculate Mean Squared Error
trainError = mean_squared_error(yTrain, yTrainPred)
valError = mean_squared_error(yVal, yValPred)

# Calculate R² Value
trainRSquared = r2_score(yTrain, yTrainPred)
valRSquared = r2_score(yVal, yValPred)

# Print Final Results
print(f"Training Error (MSE): {trainError}")
print(f"Validation Error (MSE): {valError}")
print(f"R squared for Training Set: {trainRSquared}")
print(f"R squared for Validation Set: {valRSquared}")

# Save Graphs
if not os.path.exists('results'):
    os.makedirs('results')

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
plt.savefig('results/comparisonPlotSklearn.png')
plt.close()
