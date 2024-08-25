import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Ensure the graphs directory exists
if not os.path.exists('graphs'):
    os.makedirs('graphs')

# Process Data Set
dataset = pd.read_csv('./data/SeoulBikeData.csv')
dataset = dataset[['Hour', 'Temperature', 'Seasons', 'Holiday', 'Functioning Day', 'Rented Bike Count']]

# Convert object values
dataset_aux = []
for i in range(dataset.shape[0]):
    row_aux = dataset.iloc[i].copy()
    
    # Convert the Season
    if row_aux['Seasons'] == 'Spring':
         row_aux['Seasons'] = 0
    elif row_aux['Seasons'] == 'Summer':
         row_aux['Seasons'] = 2
    elif row_aux['Seasons'] == 'Autumn':
         row_aux['Seasons'] = 3
    else: row_aux['Seasons'] = 4

    # Convert the Holiday
    if row_aux['Holiday'] == 'Holiday':
        row_aux['Holiday'] = 1
    else:
        row_aux['Holiday'] = 0

    # Convert Functioning Day
    if row_aux['Functioning Day'] == 'Yes':
        row_aux['Functioning Day'] = 1
    else:
        row_aux['Functioning Day'] = 0

    dataset_aux.append(row_aux)

dataset = pd.DataFrame(dataset_aux)

# Save feature plots
features = ['Hour', 'Temperature', 'Seasons', 'Holiday', 'Functioning Day']
for feature in features:
    plt.figure(figsize=(10, 6))
    sns.histplot(dataset[feature], bins=30, kde=False, color='skyblue')
    plt.title(f'Distribution of {feature}')
    plt.xlabel(feature)
    plt.ylabel('Count')
    plt.savefig(f'graphs/{feature}_distribution.png')
    plt.close()

# Save Rented Bike Count as bar plot
plt.figure(figsize=(12, 8))
sns.barplot(x=dataset.index, y=dataset['Rented Bike Count'], color='lightgreen')
plt.title('Rented Bike Count Over Time')
plt.xlabel('Time')
plt.ylabel('Rented Bike Count')
plt.savefig('graphs/rented_bike_count_bars.png')
plt.close()
