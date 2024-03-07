import pandas as pd
import numpy as np
import readData
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import tree
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import export_text

import sklearn.metrics as sm
import graphviz
import sklearn.preprocessing as preprocessing



data = readData.loadData('../Data/weight_loss_dataset.csv', 'csv')
pd.set_option('display.float_format', lambda x: '%.3f' % x)

weightLoss = []
for index in range(len(data)):
    weightLoss.append(data.loc[index, 'Starting_Weight_KG'] - data.loc[index, 'End_Weight_KG'])

data = data.assign(Weight_Loss=weightLoss)

# I need to change gender to a numerical value and change intensity to a numerical value (Low = 0, Medium = 1, High = 2)
data['Gender'].replace({'Male': 0, 'Female': 1}, inplace=True)
data['Intensity'].replace({'Low': 0, 'Medium': 1, 'High': 2}, inplace=True)
print(data)


classification_columns = ['Duration_in_weeks', 'Training_hours_per_week', 'Intensity']
regression_columns = ['Duration_in_weeks', 'Training_hours_per_week', 'Intensity', 'Weight_Loss']

# Create a categorical variable for classification
data['Class_Label'] = pd.cut(data['Weight_Loss'], bins=[-np.inf, 0, np.inf], labels=[0, 1], right=False)

# Classification
X_class = data[classification_columns].values
y_class = data['Class_Label'].astype(int).values

# Regression
X_reg = data[regression_columns].values
y_reg = data['Weight_Loss'].values
y_reg = data['Weight_Loss'].values

def randomForestClassifier():
    set_prop = 0.2
    seed = 5
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X_class, y_class, test_size=set_prop, random_state=seed)
    parameters = {'max_depth': 5, 'n_estimators': 100}
    clf = RandomForestClassifier(**parameters)
    clf.fit(X_train, y_train)
    testp = clf.predict(X_test)
    print("Accuracy for classification using decisionTree: ", accuracy_score(y_test, testp))



def randomForestRegressor():
    set_prop = 0.2
    seed = 5
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X_reg, y_reg, test_size=set_prop, random_state=seed)
    parameters = {'max_depth': 5, 'n_estimators': 100}
    regressor = RandomForestRegressor(**parameters)
    regressor.fit(X_train, y_train)
    testp = regressor.predict(X_test)
    mse = sm.mean_squared_error(y_test, testp)
    print("Mean Squared Error for regression using RandomForestRegressor: ", mse)


randomForestClassifier()
randomForestRegressor()

