import numpy as np
import pandas as pd
import seaborn as sb
from matplotlib import pyplot as plt

# Process Data Set
# Load and clean data
dataSet = pd.read_csv('./data/SeoulBikeData.csv')

#Columns
# Date - Date - Date of capture (year-month-day) - ** index **
# Hour - Integer - Hour of he day - ** index **
# Rented Bike Count - Integer - Count of bikes rented at each hour - **target**
# Temperature - Continuous - C - Temperature in Celsius
# Humidity - Integer - % - Humidity of the environment
# Wind speed - Continuous - m/s 
# Visibility - Integer - 10m
# Dew point temperature - Continuous - C
# Solar Radiation - Continuous - Mj/m2
# Rainfall - Integer - mm
# Snowfall - Integer - cm
# Seasons - Categorical - Current season of the year (Winter, Spring, Summer, Autumn)
# Holiday - Binary - Holiday/No holiday

# Rename columns
dataSet = dataSet.rename(
    columns={
        'Date' : 'date',
        'Rented Bike Count' : 'rentedBikeCount',
        'Hour' : 'hour',
        'Temperature' : 'temperature',
        'Humidity(%)' : 'humidity',
        'Wind speed (m/s)' : 'windSpeed',
        'Visibility (10m)' : 'visibility',
        'Dew point temperature' : 'dewPoint',
        'Solar Radiation (MJ/m2)' : 'solarRadiation',
        'Rainfall(mm)' : 'rainfall',
        'Snowfall (cm)' : 'snowfall',
        'Seasons' : 'season',
        'Holiday' : 'holiday',
        'Functioning Day' : 'functioningDay'
    }
)

#search for duplicates
dataSet = dataSet.drop_duplicates()

#Drop innecesary columns
dataSet = dataSet.drop(columns=['functioningDay'])

#Separate the values of date for only save the month
dataSet['date'] = pd.to_datetime(dataSet['date'], format='%d/%m/%Y')
dataSet['month'] = dataSet['date'].dt.month
dataSet = dataSet.drop(columns=['date'])

#Convert season string in int value
dataSet = pd.get_dummies(dataSet, columns=['season'])
newColumns = [col for col in dataSet.columns if col.startswith('season_')]
dataSet[newColumns] = dataSet[newColumns].astype(int)

dataSet['holiday'] = dataSet['holiday'].map({'No Holiday': 0, 'Holiday': 1})

# Convert categorical values to numeric
print(dataSet.shape)
print(dataSet)

dataSet.to_csv('./data/SeoulBikeDataProcessed.csv', index=False)
