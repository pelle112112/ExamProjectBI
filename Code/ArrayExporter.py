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

def returnExerciseArrays(filePath):
    firstDF = pd.read_csv(filePath)
    typesOfExercises = ['Stationary cycling', 'Calisthenics', 'Circuit training', 'Weight lifting', 'Stair machine', 'Rowing machine', 'Ski machine', 'Aerobics', 'Stretching', 'Mild stretching', 'Instructing aerobic class', 'Running', 'Martial arts']

    newGymExercises = firstDF[firstDF['Activity, Exercise or Sport (1 hour)'].str.contains('|'.join(typesOfExercises))]
    highExercises = newGymExercises[newGymExercises['Activity, Exercise or Sport (1 hour)'].str.contains('vigorous|fast')]
    mediumExercises = newGymExercises[newGymExercises['Activity, Exercise or Sport (1 hour)'].str.contains('moderate|general')] 
    lowExercises = newGymExercises[newGymExercises['Activity, Exercise or Sport (1 hour)'].str.contains('light|slow|minimal')]
    return highExercises, mediumExercises, lowExercises