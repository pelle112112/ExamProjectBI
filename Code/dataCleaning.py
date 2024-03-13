import Code.readData as rd
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
from sklearn.metrics import mean_squared_error



# Deciding on which datasets to use
exercise_datasetDF = pd.read_csv('../Data/exerciseDataset.csv')
megaGym_datasetDF = rd.loadData('../Data/megaGymDataset.csv', 'csv')
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
# We might do it later.

# Lets check for correlation
numericExerciseDF.hist(figsize=(15, 10))
plt.tight_layout()
plt.show()
plt.figure(figsize=(15, 10))
sns.heatmap(numericExerciseDF.corr(), annot=True)
plt.show()

# On the correlation map, we can see that the the exercise dataset holds a lot of correlation between the different exercises, which is expected, since the exercises are similar in nature, and should be correlated
# The correlation between the body weights and amount of calories burned are 1=1. Can this be true?

# We need to find the exercises the users can actually do at the gym
exercise_types = ['Stationary cycling', 'Calisthenics', 'Circuit training', 'Weight lifting', 'Stair machine', 'Rowing machine', 'Ski machine', 'Aerobics', 'Stretching', 'Mild stretching', 'Instructing aerobic class', 'Running', 'Martial arts']

gymExercises = exercise_datasetDF[exercise_datasetDF['Activity, Exercise or Sport (1 hour)'].str.contains('|'.join(exercise_types))]

# Now we have the exercises that the users can do at the gym called gymExercises
# Lets visualize the exercises that the users can do at the gym and the amount of calories they burn
plt.figure(figsize=(15, 10))
sns.barplot(x='Calories per kg', y='Activity, Exercise or Sport (1 hour)', data=gymExercises)
plt.show()

# How can we use the megagym dataset to create a 1 week workout plan for the users?
# We need to find out how trained the users are, and then we can use the megagym dataset to create a workout plan for them

def returnExercisesByLevel(level):
   exercises = megaGym_datasetDF[megaGym_datasetDF['Level'] == level]
   return exercises

beginnerExercises = returnExercisesByLevel('Beginner')
print(beginnerExercises)


# What we want to do now is create a model that can predict the amount of time the user needs to spend at the gym, to reach their goal
# We will use the exercise dataset to create this model
# We will use the amount of calories burned, the body weight and the type of exercise to predict the amount of time the user needs to spend at the gym
# We will use a linear regression model to predict the amount of time the user needs to spend at the gym
# The bodyweight will come as input from the user, and the type of exercise will be chosen by the user
# The amount of calories burned will be calculated by the model, and the amount of time the user needs to spend at the gym will be predicted by the model


'''

def train_calories_burned_model(dataset):
    # Assuming 'Activity' is the activity name, and the rest are weights and calories burned
    X = dataset[['130 lb', '155 lb', '180 lb', '205 lb']]
    y = dataset['Calories per kg']

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a linear regression model
    model = LinearRegression()

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    print(f'Mean Squared Error: {mse}')

    # Visualize the predictions
    plt.scatter(X_test['130 lb'], y_test, color='black', label='Actual')
    plt.scatter(X_test['130 lb'], y_pred, color='blue', label='Predicted')
    plt.xlabel('Weight (130 lb)')
    plt.ylabel('Calories per kg')
    plt.legend()
    plt.show()

    return model

# Train the model
calories_burned_model = train_calories_burned_model(exercise_datasetDF)
print(calories_burned_model)

'''

# Find all the intensity exercises in the gymExercises dataset. We know they are either listed as: vigorous or fast
# We then find the average amount of calories burned per kg for these exercises
# We have to convert the values from kg to lbs. 1 kg = 2.2 lbs
highIntensityExercises = gymExercises[gymExercises['Activity, Exercise or Sport (1 hour)'].str.contains('vigorous|fast')]
print(highIntensityExercises)
highAverage = 0
for i in highIntensityExercises['Activity, Exercise or Sport (1 hour)']:
    highAverage = highIntensityExercises['Calories per kg'].mean()
highAverage=highAverage*2.2
print(highAverage)

mediumIntensityExercises = gymExercises[gymExercises['Activity, Exercise or Sport (1 hour)'].str.contains('moderate|general')]
print(mediumIntensityExercises)
mediumAverage = 0
for i in mediumIntensityExercises['Activity, Exercise or Sport (1 hour)']:
    mediumAverage = mediumIntensityExercises['Calories per kg'].mean()
mediumAverage = mediumAverage*2.2
print(mediumAverage)

lowIntensityExercises = gymExercises[gymExercises['Activity, Exercise or Sport (1 hour)'].str.contains('light|slow|minimal')]
print(lowIntensityExercises)
lowAverage = 0
for i in lowIntensityExercises['Activity, Exercise or Sport (1 hour)']:
    lowAverage = lowIntensityExercises['Calories per kg'].mean()

lowAverage = lowAverage*2.2
print(lowAverage)


