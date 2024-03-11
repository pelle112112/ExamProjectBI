import dataCleaning as dc
import numpy as np
import pandas as pd

import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import fcluster

from sklearn.cluster import AgglomerativeClustering
from sklearn import metrics

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

trainingData = dc.gymExercises.copy()
print(trainingData)

# Remove the Martial Arts data
hcData = trainingData[~trainingData['Activity, Exercise or Sport (1 hour)'].str.contains('Martial arts')]

X = hcData.iloc[:, 2:].values
print (X)

linkageMatrix = sch.linkage(X, method='ward')

# making a  Dendogram
plt.figure(figsize=(20, 10))

dendogram = sch.dendrogram(linkageMatrix)
plt.title('Dendogram')
plt.xlabel('Exercise Id')
plt.ylabel('Euclidean distances')
plt.show()

# Automatic choose the number of clusters
#First we choose the max distance between cluster
heightCutoff = 100

# Then we use the fcluster function to get the cluster labels
clusterLabels = fcluster(linkageMatrix, heightCutoff, criterion='distance')

n_clusters = len(set(clusterLabels))
print(f'Number of clusters: {n_clusters}')

model = AgglomerativeClustering(n_clusters, linkage='ward')
model.fit(X)

labels = model.labels_

Y = model.fit_predict(X)
print (Y)

plt.scatter(X[:, 0], X[:, 0], s=50, c=Y, cmap='viridis')
plt.title('Clusters of exercises')
plt.xlabel('X1')
plt.ylabel('X2')

plt.show()

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:,0], X[:,1], X[:,2], marker='o', s=30, c=Y, cmap='viridis')

plt.show()

silhouette_score = metrics.silhouette_score(X, Y)
print(f'Silhouette score: {silhouette_score}')