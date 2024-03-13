import joblib
import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder as le
from Code.ArrayExporter import returnExerciseArrays
st.set_page_config(page_title="Training Program Calculator", layout="wide")

st.title("Training Program Calculator")
st.sidebar.header("Training Program Calculator", divider='rainbow')

high, med, low = returnExerciseArrays('./Data/exercise_dataset.csv')

duration = st.number_input('How many weeks are you training?', 1, 20)
intensity = st.selectbox("How intense do you want your training to be?", ['High', 'Medium', 'Low'])
caloriesGoal = st.number_input('How many calories do you want to burn per week?', 500, 10000)
weight = st.number_input('What is your current weight in kg?', 60, 200)
numericIntensity = None
if intensity == 'High':
    numericIntensity = 0
elif intensity == 'Medium':
    numericIntensity = 1
else:
    numericIntensity = 2    
        
data = {'Duration_in_weeks': [duration],
        'Intensity': [numericIntensity],
        'Calories_goal': [caloriesGoal],
        'Starting_Weight_KG': [weight]}

# I need to create function which calculates the best training program based on the input data
def calculateTrainingProgram():
    # Get the input data
    duration = data['Duration_in_weeks'][0]
    calories_goal = data['Calories_goal'][0]
    intensity = data['Intensity'][0]
    starting_weight = data['Starting_Weight_KG'][0]
    starting_weight = starting_weight * 2.20462  # Convert to pounds
        
    # Determine the array of exercises based on intensity
    if intensity == 0:  # High Intensity
        exercises = pd.concat([high, high, high])
        
    elif intensity == 1:  # Medium Intensity
        exercises = pd.concat([med, med, med])
    else:  # Low Intensity
        exercises = pd.concat([low, low, low])
    
    weightBasedExercises = exercises
    # Find the closest weight value in the exercise dataset between 130, 155, 180, 205
    weightOne, weightTwo, weightThree, weightFour = 130, 155, 180, 205
    weightDifference = []
    weightDifference.append(abs(starting_weight - weightOne))
    weightDifference.append(abs(starting_weight - weightTwo))
    weightDifference.append(abs(starting_weight - weightThree))
    weightDifference.append(abs(starting_weight - weightFour))
    closestWeight = weightDifference.index(min(weightDifference))

    if closestWeight == 0:
        starting_weight = weightOne
        weightBasedExercises.drop('155 lb', axis=1, inplace=True)
        weightBasedExercises.drop('180 lb', axis=1, inplace=True)
        weightBasedExercises.drop('205 lb', axis=1, inplace=True)
        weightBasedExercises.drop('Calories per kg', axis=1, inplace=True)
    elif closestWeight == 1:
        starting_weight = weightTwo
        weightBasedExercises.drop('130 lb', axis=1, inplace=True)
        weightBasedExercises.drop('180 lb', axis=1, inplace=True)
        weightBasedExercises.drop('205 lb', axis=1, inplace=True)
        weightBasedExercises.drop('Calories per kg', axis=1, inplace=True)
    elif closestWeight == 2:
        starting_weight = weightThree
        weightBasedExercises.drop('130 lb', axis=1, inplace=True)
        weightBasedExercises.drop('155 lb', axis=1, inplace=True)
        weightBasedExercises.drop('205 lb', axis=1, inplace=True)
        weightBasedExercises.drop('Calories per kg', axis=1, inplace=True)
    else:
        starting_weight = weightFour
        weightBasedExercises.drop('130 lb', axis=1, inplace=True)
        weightBasedExercises.drop('155 lb', axis=1, inplace=True)
        weightBasedExercises.drop('180 lb', axis=1, inplace=True)
        weightBasedExercises.drop('Calories per kg', axis=1, inplace=True)

    
    caloriesLeft = calories_goal


    trainingProgram = []
    # Calculate a one week training program
    
    match starting_weight:
        case 130:
            for exercise in weightBasedExercises['130 lb']:
                if caloriesLeft > exercise/2:
                    trainingProgram.append(exercise)
                    caloriesLeft = caloriesLeft - exercise
                    print("Calories left: ", caloriesLeft)
                    print("Exercise: ", exercise)
            trainingProgram = weightBasedExercises[weightBasedExercises['130 lb'].isin(trainingProgram)]
            trainingProgram.insert(2, "Minutes per week", 60)
        case 155:
            for exercise in weightBasedExercises['155 lb']:
                if caloriesLeft > exercise/2:
                    trainingProgram.append(exercise)
                    caloriesLeft = caloriesLeft - exercise
                    print("Calories left: ", caloriesLeft)
                    print("Exercise: ", exercise)
            trainingProgram = weightBasedExercises[weightBasedExercises['155 lb'].isin(trainingProgram)]
            trainingProgram.insert(2, "Minutes per week", 60)
        case 180:
            for exercise in weightBasedExercises['180 lb']:
                if caloriesLeft > exercise/2:
                    trainingProgram.append(exercise)
                    caloriesLeft = caloriesLeft - exercise
                    print("Calories left: ", caloriesLeft)
                    print("Exercise: ", exercise)
            trainingProgram = weightBasedExercises[weightBasedExercises['180 lb'].isin(trainingProgram)]
            trainingProgram.insert(2, "Minutes per week", 60)
        case 205:
            for exercise in weightBasedExercises['205 lb']:
                if(caloriesLeft > exercise/2):
                        trainingProgram.append(exercise)
                        caloriesLeft= caloriesLeft- exercise
                        print("Calories left: ", caloriesLeft) 
                        print("Exercise: ", exercise)
            trainingProgram = weightBasedExercises[weightBasedExercises['205 lb'].isin(trainingProgram)]
            trainingProgram.insert(2, "Minutes per week", 60)

    
    return trainingProgram, caloriesLeft

# Call the function when the button is clicked
makePrediction = st.button('Make prediction')
if makePrediction:
    exercises, caloriesLeft = calculateTrainingProgram()
    st.write("Recommended Exercises:")
    st.write(exercises)
    st.success("Do these exercises 1 hour each a week to reach your goal!")
    st.write("Calories burned per week: ", caloriesGoal-caloriesLeft)

    
   
