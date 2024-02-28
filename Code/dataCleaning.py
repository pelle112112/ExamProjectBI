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
numericExerciseDF = exercise_datasetDF._get_numeric_data()
print(exercise_datasetDF)


numericExerciseDF.plot(kind='box', figsize=(20, 10))
plt.tight_layout()
plt.show()

# Checking for outliers
sc = StandardScaler()
StandardDF = sc.fit_transform(numericExerciseDF)
StandardDF = pd.DataFrame(StandardDF, columns=numericExerciseDF.columns)
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
numericExerciseDF.hist(figsize=(15, 10))
plt.tight_layout()
plt.show()
plt.figure(figsize=(15, 10))
sns.heatmap(numericExerciseDF.corr(), annot=True)
plt.show()

# On the correlation map, we can see that the the exercise dataset holds a lot of correlation between the different exercises, which is expected, since the exercises are similar in nature, and should be correlated
# The correlation between the body weights and amount of calories burned are 1=1

# We need to find the exercises the users can actually do at the gym
exercise_types = ['Stationary cycling', 'Calisthenics', 'Circuit training', 'Weight lifting', 'Stair machine', 'Rowing machine', 'Ski machine', 'Aerobics', 'Stretching', 'Mild stretching', 'Instructing aerobic class', 'Running', 'Martial arts']

gymExercises = exercise_datasetDF[exercise_datasetDF['Activity, Exercise or Sport (1 hour)'].str.contains('|'.join(exercise_types))]

# Now we have the exercises that the users can do at the gym called gymExercises
# Lets visualize the exercises that the users can do at the gym and the amount of calories they burn
plt.figure(figsize=(15, 10))
sns.barplot(x='Calories per kg', y='Activity, Exercise or Sport (1 hour)', data=gymExercises)
plt.show()

