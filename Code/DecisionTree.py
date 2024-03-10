import pandas as pd
import numpy as np
import readData
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.naive_bayes import GaussianNB


# Your existing code for loading and preprocessing data
data = readData.loadData('../Data/weight_loss_dataset.csv', 'csv')
pd.set_option('display.float_format', lambda x: '%.3f' % x)

weightLoss = []
for index in range(len(data)):
    weightLoss.append(data.loc[index, 'Starting_Weight_KG'] - data.loc[index, 'End_Weight_KG'])

data = data.assign(Weight_Loss=weightLoss)

# Convert gender and intensity to numerical values
data['Gender'].replace({'Male': 0, 'Female': 1}, inplace=True)
data['Intensity'].replace({'Low': 0, 'Medium': 1, 'High': 2}, inplace=True)

# Regression columns
regression_columns = ['Duration_in_weeks', 'Training_hours_per_week', 'Intensity', 'Weight_Loss']

# Classification label creation (0 for no weight loss, 1 for weight loss)
#data['Class_Label'] = pd.cut(data['Weight_Loss'], bins=[-np.inf, 0, np.inf], labels=[0, 1], right=False)
# Changing class label to predict the amount of weight loss
data['Class_Label'] = data['Weight_Loss']


# Feature and target for classification
X_class = data[regression_columns[:-1]].values
y_class = data['Class_Label'].astype(int).values

# Feature and target for regression
X_reg = data[regression_columns[:-1]].values
y_reg = data['Weight_Loss'].values

def randomForestClassifier():
    set_prop = 0.2
    seed = 5
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X_class, y_class, test_size=set_prop, random_state=seed)
    parameters = {'max_depth': 5, 'n_estimators': 100}
    clf = RandomForestClassifier(**parameters)

    # Perform cross-validation
    scores = model_selection.cross_val_score(clf, X_class, y_class, cv=5)
    print("Cross-validated Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    # Fit the model on the training data
    clf.fit(X_train, y_train)

    # Evaluate on the test set
    testp = clf.predict(X_test)
    print("Accuracy for classification using RandomForestClassifier: ", accuracy_score(y_test, testp))

    # Display confusion matrix
    confusion_mat = confusion_matrix(y_test, testp)
    print("Confusion Matrix:")
    print(confusion_mat)

    plt.imshow(confusion_mat, interpolation='nearest')
    plt.title('Confusion matrix')
    plt.colorbar()
    ticks = np.arange(14)
    plt.xticks(ticks, ticks)
    plt.yticks(ticks, ticks)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


def randomForestRegressor():    
    set_prop = 0.2
    seed = 7
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X_reg, y_reg, test_size=set_prop, random_state=seed)
    parameters = {'max_depth': 5, 'n_estimators': 100}
    regressor = RandomForestRegressor(**parameters)
    regressor.fit(X_train, y_train)
    testp = regressor.predict(X_test)
    mse = np.mean((y_test - testp) ** 2)  # Mean Squared Error for regression
    print("Mean Squared Error for regression using RandomForestRegressor: ", mse)
    plt.scatter(X_test[:, 1], y_test, color='black', label='Actual')
    plt.scatter(X_test[:, 1], testp, color='blue', label='Predicted')
    plt.xlabel('Training hours per week')
    plt.ylabel('Weight Loss')
    plt.legend()
    plt.show()

# Call the functions
randomForestClassifier()
randomForestRegressor()


# Naive Bayes Classifier
def naiveBayesClassifier():
    set_prop = 0.2
    seed = 5
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X_class, y_class, test_size=set_prop, random_state=seed)
    clf = GaussianNB()

    # Perform cross-validation
    scores = model_selection.cross_val_score(clf, X_class, y_class, cv=5)
    print("Cross-validated Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    clf.fit(X_train, y_train)
    testp = clf.predict(X_test)
    print("Accuracy for classification using Naive Bayes Classifier: ", accuracy_score(y_test, testp))

    # Display confusion matrix
    confusion_mat = confusion_matrix(y_test, testp)
    print("Confusion Matrix:")
    print(confusion_mat)

    plt.imshow(confusion_mat, interpolation='nearest')
    plt.title('Confusion matrix')
    plt.colorbar()
    ticks = np.arange(14)
    plt.xticks(ticks, ticks)
    plt.yticks(ticks, ticks)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


naiveBayesClassifier()