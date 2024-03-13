import dataCleaning as dc

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from yellowbrick.cluster import SilhouetteVisualizer

trainingData = dc.gymExercises.copy()
print(trainingData.columns)
print(trainingData.head())

# Remove the Martial Arts data
processesedTrainingData = trainingData[~trainingData['Activity, Exercise or Sport (1 hour)'].str.contains('Martial arts')]

print(processesedTrainingData)

# Ad a new intensity categoory
'''
intensityCategory = {
    'very light': lambda x: x < 0.5,
    'light': lambda x: (0.5 <= x) & (x < 1),
    'moderate': lambda x: (1 <= x) & (x < 2),
    'vigorous': lambda x: (2 <= x) & (x < 3),
    'very vigorous': lambda x: x >= 3,
}

processesedTrainingData['Intensity'] = None

for category, condition in intensityCategory.items():
    processesedTrainingData.loc[condition(processesedTrainingData['Calories per kg']), 'Intensity'] = category

print(processesedTrainingData[['Activity, Exercise or Sport (1 hour)', 'Calories per kg', 'Intensity']])
print (processesedTrainingData)

intensityMapping = {
    'very light': 1,
    'light': 2,
    'moderate': 3,
    'vigorous': 4,
    'very vigorous': 5,
}

processesedTrainingData['Intensity'] = processesedTrainingData['Intensity'].map(intensityMapping)

print(processesedTrainingData)
'''
intensities = []
for activity in processesedTrainingData['Activity, Exercise or Sport (1 hour)']:
    if re.search(r'light|slow|minimal', activity):
        intensities.append(2)
    elif re.search(r'moderate|general', activity):
        intensities.append(1)
    elif re.search(r'vigorous|fast', activity):
        intensities.append(0)
    else:
        intensities.append(None) 

processesedTrainingData['Intensity'] = intensities
print(processesedTrainingData)

processesedTrainingData = processesedTrainingData[processesedTrainingData['Intensity'].notna()]

le = LabelEncoder()
processesedTrainingData['Intensity'] = le.fit_transform(processesedTrainingData['Intensity'])
print(processesedTrainingData)

# Selected the numeric columns for clustering
# numeric_columns = ['130 lb', '155 lb', '180 lb', '205 lb', 'Calories per kg']
numeric_columns = ['Calories per kg', 'Intensity']
numeric_data = processesedTrainingData[numeric_columns]

# Standardize the data
#scaler = StandardScaler()
#scaled_data = scaler.fit_transform(numeric_data)


silScore = []

for i in range(2, 11):
    kmeans = KMeans(n_clusters=i, random_state=0).fit(numeric_data)
    clusterLabels = kmeans.labels_
    silhouette_avg = silhouette_score(numeric_data, clusterLabels)
    silScore.append(silhouette_avg)


plt.plot(range(2, 11), silScore)
plt.title('Silhouette Analysis')
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette Score')
plt.show()

# The optimal number of clusters
optimalK = 3

kmeans = KMeans(n_clusters=optimalK, init='k-means++', max_iter=300, n_init=10, random_state=0)
kmeans.fit(numeric_data[['Calories per kg', 'Intensity']])

test = kmeans.predict([[5.601, 1]])
print(test)

clusterLabels = kmeans.labels_

silhouette_avg = silhouette_score(numeric_data, clusterLabels)

print('Silhouette score = ', silhouette_avg)


X = numeric_data[['Intensity']]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=optimalK, random_state=0)
kmeans.fit(X)
y = kmeans.predict(X)

# Plotting the clusters
for i in range(optimalK):
    cluster = numeric_data[y == i]

    print("Cluster ", i, ":", cluster.shape)

    plt.scatter(cluster['Intensity'], cluster['Calories per kg'], label=f'Cluster {i}')

plt.legend()
plt.xlabel('Intensity')
plt.ylabel('Calories per kg')
plt.grid(True)
plt.show()

model = KMeans(n_clusters=optimalK, n_init=10)
visualizer = SilhouetteVisualizer(model, colors='yellowbrick')

# Fit the visualizer
visualizer.fit(numeric_data)

# Show the visualizer
visualizer.show()