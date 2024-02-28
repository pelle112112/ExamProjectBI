import readData
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import sklearn.metrics as sm
from sklearn.preprocessing import PolynomialFeatures
import numpy as np



# Deciding on which datasets to use
exercise_datasetDF = readData.loadData('../Data/exercise_dataset.csv', 'csv')
megaGym_datasetDF = readData.loadData('../Data/megaGymDataset.csv', 'csv')
#print(exercise_datasetDF)
#print(megaGym_datasetDF)


# Data wrangling of both DF
# The exercise dataset doesnt hold any missing values
# The megaGym Dataset holds lots of missing values, in form of descriptions, ratings and rating descriptions
print(exercise_datasetDF.isnull().sum())
print(megaGym_datasetDF.isnull().sum())

# Lets remove the desc, rating and ratingdesc columns from the megagym set
megaGym_datasetDF.drop('Desc', axis=1, inplace=True)
megaGym_datasetDF.drop('Rating', axis=1, inplace=True)
megaGym_datasetDF.drop('RatingDesc', axis=1, inplace=True)
print(megaGym_datasetDF)
print(megaGym_datasetDF.describe)
print(exercise_datasetDF.describe)

# Removing non numerical columns for the exercise dataset
nonNumericExerciseDF = exercise_datasetDF._get_numeric_data()
print(exercise_datasetDF)


nonNumericExerciseDF.plot(kind='box', figsize=(20, 10))
plt.tight_layout()
plt.show()

# Checking for outliers
sc = StandardScaler()
StandardDF = sc.fit_transform(nonNumericExerciseDF)
StandardDF = pd.DataFrame(StandardDF, columns=nonNumericExerciseDF.columns)
StandardDF.plot(kind='box', figsize=(20, 10))
plt.tight_layout()
plt.show()

# Pinpointing the outliers
outliers = []
for column in StandardDF.columns:
    Q1 = StandardDF[column].quantile(0.25)
    Q3 = StandardDF[column].quantile(0.75)
    IQR = Q3 - Q1
    outlier = StandardDF[(StandardDF[column] < (Q1 - 1.5 * IQR)) | (StandardDF[column] > (Q3 + 1.5 * IQR))]
    outliers.append(outlier)
outliers = pd.concat(outliers)
print(outliers)

# We wont remove the outliers, since they are a real exercise, and they are adding value to the dataset, by being a productive exercise for our users

# Lets check for correlation
nonNumericExerciseDF.hist(figsize=(15, 10))
plt.tight_layout()
plt.show()
plt.figure(figsize=(15, 10))
sns.heatmap(nonNumericExerciseDF.corr(), annot=True)
plt.show()

# On the correlation map, we can see that the the exercise dataset holds a lot of correlation between the different exercises, which is expected, since the exercises are similar in nature, and should be correlated
# The correlation between the body weights and amount of calories burned are 1=1

