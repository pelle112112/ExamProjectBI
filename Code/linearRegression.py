import pandas as pd
import numpy as np
import readData
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import sklearn.metrics as sm


data = readData.loadData('../Data/weight_loss_dataset.csv', 'csv')
pd.set_option('display.float_format', lambda x: '%.3f' % x)

# Checking for null values and which datatypes are in the dataframe
print(data.isnull().sum())
print(data.dtypes)

# Descriptive statistics
print(data.describe())

# Plotting histogram to get an overview of the data
data.hist()
plt.tight_layout()
plt.show()

# Inserting a column that holds the differnce between starting weight and end weight.
weightLoss = []
for index in range(len(data)):
    weightLoss.append(data.loc[index, 'Starting_Weight_KG'] - data.loc[index, 'End_Weight_KG'])

data = data.assign(Weight_Loss=weightLoss)

print(data.describe())

# Looking at weight loss across different intensities. 
intensityData = data.drop(['Gender', 'Starting_Weight_KG', 'End_Weight_KG'], axis=1)
print(intensityData)
print(intensityData.describe())
intensityData = intensityData.groupby(by='Intensity')
intensityData.mean().plot(kind='bar')
plt.tight_layout()
plt.show()

# Making heatmap to show correlation between features and weight loss
le = preprocessing.LabelEncoder()
heatmapData = data.copy()
heatmapData['Gender'] = le.fit_transform(heatmapData['Gender'])
heatmapData['Intensity'] = le.fit_transform(heatmapData['Intensity'])
plt.figure(figsize=(10, 10))
sns.heatmap(heatmapData.corr(), annot=True)
plt.tight_layout()
plt.show()
# Highest correlation to weight loss is Training_hours_per_week

# Making scatterplot to get an idea if the data is suitable for linear regression
plt.figure(figsize=(10, 10))
plt.ylabel('Weight Loss')
plt.xlabel('Training hours per week')
plt.scatter(data['Training_hours_per_week'].values.reshape(-1, 1), data['Weight_Loss'].values.reshape(-1, 1))
plt.tight_layout()
plt.show()
# Data is suitable for linear regression, but might need some optimisation.

# Training a model using regular linear regression
X_train, X_test, y_train, y_test = train_test_split(data['Training_hours_per_week'], data['Weight_Loss'], random_state=5, test_size=0.2)
lineModel = LinearRegression()
lineModel.fit(X_train, y_train)

linearPredictions = lineModel.predict(X_test)
RMSE = np.sqrt(sm.mean_squared_error(y_test, linearPredictions))
R_Squared = sm.r2_score(y_test, linearPredictions)
print(RMSE)
print(R_Squared)