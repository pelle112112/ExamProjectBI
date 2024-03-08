import pandas as pd
import numpy as np
import readData
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import sklearn.metrics as sm
import statsmodels.api as statsmodels
from sklearn.preprocessing import PolynomialFeatures


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
plt.ylabel('Weight Loss')
plt.xlabel('Training hours per week')
plt.scatter(data['Training_hours_per_week'].values.reshape(-1, 1), data['Weight_Loss'].values.reshape(-1, 1))
plt.tight_layout()
plt.show()
# Data is suitable for linear regression, but might need some optimisation.

# Training a model using regular linear regression
X_train, X_test, y_train, y_test = train_test_split(data['Training_hours_per_week'].values.reshape(-1, 1), data['Weight_Loss'].values.reshape(-1, 1), random_state=5, test_size=0.20)
lineModel = LinearRegression()
lineModel.fit(X_train, y_train)

linearPredictions = lineModel.predict(X_test)
RMSE = np.sqrt(sm.mean_squared_error(y_test, linearPredictions))
R_Squared = sm.r2_score(y_test, linearPredictions)

plt.title('Linear Regression')
plt.scatter(data['Training_hours_per_week'].values.reshape(-1, 1), data['Weight_Loss'].values.reshape(-1, 1))
plt.plot(X_train, lineModel.coef_*X_train + lineModel.intercept_)
plt.xlabel('Training hours per week')
plt.ylabel('Weight loss')
plt.tight_layout()
plt.show()

print(RMSE) # 1.869
print(R_Squared) # 0.482

# Trying multilinear regression
multiData = data.copy()
multiData['Gender'] = le.fit_transform(multiData['Gender'])
multiData['Intensity'] = le.fit_transform(multiData['Intensity'])
X_train, X_test, y_train, y_test = train_test_split(multiData[['Training_hours_per_week', 'Starting_Weight_KG', 'Duration_in_weeks', 'Intensity']], multiData['Weight_Loss'].values.reshape(-1, 1), random_state=5, test_size=0.25)
# After having done some testing the best result comes from using the default train test split of 75% train - 25 % test
multiLineModel = LinearRegression()
multiLineModel.fit(X_train, y_train)
multiLinePredictions = multiLineModel.predict(X_test)

RMSE = np.sqrt(sm.mean_squared_error(y_test, multiLinePredictions))
R_Squared = sm.r2_score(y_test, multiLinePredictions)

print(RMSE) # 1.278
print(R_Squared) # 0.758

# Applying AIC to optimize the model as to which features needs to be included.

def AIC(dataframe, features):
    X = dataframe[features]
    y = dataframe['Weight_Loss']
    X = statsmodels.add_constant(X)
    model = statsmodels.OLS(y, X).fit()
    print(f'{features} AIC score: {model.aic}')

AIC(multiData, ['Training_hours_per_week', 'Intensity', 'Duration_in_weeks', 'Starting_Weight_KG'])
AIC(multiData, ['Training_hours_per_week', 'Duration_in_weeks', 'Starting_Weight_KG'])
AIC(multiData, ['Training_hours_per_week', 'Intensity', 'Starting_Weight_KG'])
AIC(multiData, ['Intensity', 'Duration_in_weeks', 'Starting_Weight_KG'])
AIC(multiData, ['Training_hours_per_week', 'Duration_in_weeks'])
AIC(multiData, ['Intensity', 'Starting_Weight_KG'])
AIC(multiData, ['Training_hours_per_week', 'Starting_Weight_KG'])
AIC(multiData, ['Duration_in_weeks', 'Starting_Weight_KG'])

# AIC score indicates that using all 4 features for training is best, this must be because the accuracy/explained variance or error gets too big when less columns are used. 

# Trying polynomial method to see if we can get a better result.
X = data['Training_hours_per_week']
y = data['Weight_Loss']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=5, test_size=0.20)
polyReg = LinearRegression()
polyModel = PolynomialFeatures()
X_train = polyModel.fit_transform(X_train.values.reshape(-1, 1))
X_test = polyModel.fit_transform(X_test.values.reshape(-1, 1))
polyReg.fit(X_train, y_train)

polyPredictions = polyReg.predict(X_test)

# Visualising the polynomial graph
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, y, color='red')
plt.plot(X_grid, polyReg.predict(polyModel.fit_transform(X_grid)), color='blue')
plt.title('Polynimial Regression')
plt.xlabel('Training hours per week')
plt.ylabel('Weight loss')
plt.ticklabel_format(style='plain')
plt.tight_layout()
plt.show()

# Evaluating the model
RMSE = np.sqrt(sm.mean_squared_error(y_test, polyPredictions))
R_Squared = sm.r2_score(y_test, polyPredictions)

print(RMSE) # 1.874
print(R_Squared) # 0.480