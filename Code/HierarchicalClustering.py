import dataCleaning as dc
import kMean as km
import numpy as np
import pandas as pd

import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import pdist

from sklearn.cluster import AgglomerativeClustering
from sklearn import metrics

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

hcData = km.processesedTrainingData.copy()
print(hcData)

X = hcData.iloc[:, 2:].values
print (X)


# making a  Dendogram
plt.figure(figsize=(20, 10))

dendogram = sch.dendrogram(sch.linkage(X, method='ward'))
plt.title('Dendogram')
plt.xlabel('Exercise Id')
plt.ylabel('Euclidean distances')
plt.show()

n_clusters = 4

model = AgglomerativeClustering(n_clusters, linkage='ward')
model.fit(X)

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