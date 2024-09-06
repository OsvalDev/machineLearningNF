import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Load Data Set
dataSet = pd.read_csv('./data/SeoulBikeDataProcessed.csv')

#---------------------------------------------------------
# Data preprocessing
dataCorr = dataSet.corr()
plt.figure(figsize=(10, 10))
sns.heatmap(dataCorr, cmap='coolwarm', linewidths=0.1, annot=True, linecolor='white')
plt.savefig('charts/correlation.png')

vifData = pd.DataFrame()
vifData["feature"] = dataSet.columns
vifData["VIF"] = [variance_inflation_factor(dataSet.values, i) for i in range(len(dataSet.columns))]

dataSet = dataSet.drop(columns=['dewPoint'])

q1 = np.percentile(dataSet['rentedBikeCount'], 25)
q3 = np.percentile(dataSet['rentedBikeCount'], 75)
iqr = q3 - q1

lowerBound = q1 - 1.5 * iqr
upperBound = q3 + 1.5 * iqr

dataSet = dataSet[(dataSet['rentedBikeCount'] <= upperBound)]

plt.figure(figsize=(10, 6))
sns.histplot(dataSet['rentedBikeCount'], kde=True)
plt.title('Distribución de rentedBikeCount')
plt.hist(dataSet['rentedBikeCount'], bins=30)
plt.savefig('charts/histograma.png')

#---------------------------------------------------------
# Model fit

# Separate variables
x = dataSet.drop(columns=['rentedBikeCount'])
y = dataSet['rentedBikeCount'].values

xTrain, xTemp, yTrain, yTemp = train_test_split(x, y, test_size=0.3, random_state=42)
xVal, xTest, yVal, yTest = train_test_split(xTemp, yTemp, test_size=0.4, random_state=42)

# Scale data
scaler = RobustScaler()
xTrainScaled = scaler.fit_transform(xTrain)
xValScaled = scaler.transform(xVal)
xTestScaled = scaler.transform(xTest)

# Initialize Random Forest model
rf = RandomForestRegressor(random_state=42)

# Values for hyperparameters
# paramGrid = {
#     'max_depth': [10, 15, 20],
#     'min_samples_leaf': [4, 5, 10],
#     'min_samples_split': [10, 15, 20],
#     'n_estimators': [400, 600, 800],
# }
paramGrid = {
    'max_depth': [10],
    'min_samples_leaf': [4],
    'min_samples_split': [10],
    'n_estimators': [400],
}

# GridSearchCV to find the best hyperparameters
gridSearch = GridSearchCV(estimator=rf, param_grid=paramGrid, cv=3, n_jobs=-1, scoring='r2', verbose=2)
gridSearch.fit(xTrainScaled, yTrain)

# Extract the best model
bestRf = gridSearch.best_estimator_

# Make Predictions with Random Forest best estimator
yTrainPred = bestRf.predict(xTrainScaled)
yValPred = bestRf.predict(xValScaled)
yTestPred = bestRf.predict(xTestScaled)

# Calculate Mean Squared Error
trainError = mean_squared_error(yTrain, yTrainPred)
valError = mean_squared_error(yVal, yValPred)
testError = mean_squared_error(yTest, yTestPred)

# Calculate R² Value
trainRSquared = r2_score(yTrain, yTrainPred)
valRSquared = r2_score(yVal, yValPred)
testRSquared = r2_score(yTest, yTestPred)

# Print Final Results
print(f"Best Parameters from Grid Search: {gridSearch.best_params_}")
print(f"RandomForest Training Error (MSE): {trainError}")
print(f"RandomForest Validation Error (MSE): {valError}")
print(f"RandomForest Test Error (MSE): {testError}")
print(f"RandomForest R squared for Training Set: {trainRSquared}")
print(f"RandomForest R squared for Validation Set: {valRSquared}")
print(f"RandomForest R squared for Test Set: {testRSquared}")

print('Regularization with Bagging')

# Bagging Regressor with the best Random Forest model
baggingRegressor = BaggingRegressor(estimator=bestRf, n_estimators=10, random_state=42)

baggingRegressor.fit(xTrainScaled, yTrain)

# Make Predictions with Bagging Regressor
yTrainPred = baggingRegressor.predict(xTrainScaled)
yValPred = baggingRegressor.predict(xValScaled)
yTestPred = baggingRegressor.predict(xTestScaled)

# Calculate Mean Squared Error
trainError = mean_squared_error(yTrain, yTrainPred)
valError = mean_squared_error(yVal, yValPred)
testError = mean_squared_error(yTest, yTestPred)

# Calculate R² Value
trainRSquared = r2_score(yTrain, yTrainPred)
valRSquared = r2_score(yVal, yValPred)
testRSquared = r2_score(yTest, yTestPred)

# Print Final Results
print(f"Bagging Training Error (MSE): {trainError}")
print(f"Bagging Validation Error (MSE): {valError}")
print(f"Bagging Test Error (MSE): {testError}")
print(f"Bagging R squared for Training Set: {trainRSquared}")
print(f"Bagging R squared for Validation Set: {valRSquared}")
print(f"Bagging R squared for Test Set: {testRSquared}")

#---------------------------------------------------------
# Save Graphs
if not os.path.exists('results'):
    os.makedirs('results')

# Plot comparison of expected and predicted values
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.scatter(yTrain, yTrainPred, color='blue', alpha=0.5)
plt.plot([min(yTrain), max(yTrain)], [min(yTrain), max(yTrain)], color='red', linewidth=2)
plt.xlabel('Expected Values (Train)')
plt.ylabel('Predicted Values (Train)')
plt.title('Training: Expected / Predicted')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.scatter(yTest, yTestPred, color='green', alpha=0.5)
plt.plot([min(yTest), max(yTest)], [min(yTest), max(yTest)], color='red', linewidth=2)
plt.xlabel('Expected Values (Test)')
plt.ylabel('Predicted Values (Test)')
plt.title('Test: Expected / Predicted')
plt.grid(True)

plt.tight_layout()
plt.savefig('results/comparisonPlotBagging.png')
plt.close()

#---------------------------------------------------------
# Neural Network Model for Regression
model = Sequential()
model.add(Dense(16, input_dim=xTrainScaled.shape[1], activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(8, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1)) 

model.compile(optimizer=Adam(learning_rate=0.01), loss='mean_squared_error')

history = model.fit(xTrainScaled, yTrain, epochs=50, batch_size=32, validation_data=(xValScaled, yVal))

# Evaluate the model
loss = model.evaluate(xTestScaled, yTest)
nnTestPred = model.predict(xTestScaled).flatten()

# Calculate MSE for neural network
nnTestMse = mean_squared_error(yTest, nnTestPred)
nnTrainMse = mean_squared_error(yTrain, model.predict(xTrainScaled).flatten())
nnValMse = mean_squared_error(yVal, model.predict(xValScaled).flatten())

# Calculate R² Value for neural network
nnTrainR2 = r2_score(yTrain, model.predict(xTrainScaled).flatten())
nnValR2 = r2_score(yVal, model.predict(xValScaled).flatten())
nnTestR2 = r2_score(yTest, nnTestPred)

print(f"Neural Network Training Error (MSE): {nnTrainMse}")
print(f"Neural Network Validation Error (MSE): {nnValMse}")
print(f"Neural Network Test Error (MSE): {nnTestMse}")
print(f"Neural Network R² for Training Set: {nnTrainR2}")
print(f"Neural Network R² for Validation Set: {nnValR2}")
print(f"Neural Network R² for Test Set: {nnTestR2}")

# Plot training history
plt.figure(figsize=(8, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Neural Network Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig('results/nnLoss.png')

# Compare Random Forest, and Neural Network
print("\nComparison of Models:")
print(f"Random Forest - Test Error (MSE): {testError:.4f}")
print(f"Neural Network - Test Error (MSE): {nnTestMse:.4f}")
