import joblib
import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder as le

linearModel = joblib.load('model/bestLinearModel.pkl')
randomForestModel = joblib.load('model/randomForestClassifier.pkl')

st.set_page_config(page_title="Predictions", layout="wide")

st.title("Predictions")
st.sidebar.header("Predictions", divider='rainbow')

chosenModel = st.selectbox('**Choose the model you want to make a prediction with**', ['Linear Regression', 'Random Forest'])

match chosenModel:
    case 'Linear Regression':
        st.success('👇 You have chosen to predict with the linear regression model, please fill out the information below 👇')

        weight = st.number_input('What is your current weight in kg?', 60, 200)
        duration = st.number_input('How many weeks are you training?', 1, 52)
        hoursPerWeek = st.number_input('How many hours are you training per week?', 1, 15)
        intensity = st.selectbox("**How intense do you want your training to be?**", ['High', 'Medium', 'Low'])
        numericIntensity = None
        if intensity == 'High':
            numericIntensity = 0
        elif intensity == 'Medium':
            numericIntensity = 1
        else:
            numericIntensity = 2    
        
        data = {'Training_hours_per_week': [hoursPerWeek], 
                'Starting_Weight_KG': [weight],
                'Duration_in_weeks': [duration],
                'Intensity': [numericIntensity]}
        df = pd.DataFrame(data)
        makePrediction = st.button('Make prediction')

        if makePrediction:
            prediction = linearModel.predict(df)
            if prediction[0][0] > 0: 
                st.success(f'With a starting weight of: {weight} kg, you will most likely have lost {prediction[0][0]} kg, after having trained {duration} weeks, {hoursPerWeek} hours per week, at {intensity} intensity.')
            else:
                st.success("You won't lose any weight with what you have currently selected, try updating your training hours per week or the amount of weeks you're training.")
                
    case 'Random Forest':
        st.success('👇 You have chosen to predict with the random forest model, please fill out the information below 👇')

        weight = st.number_input('What is your current weight in kg?', 60, 200)
        duration = st.number_input('How many weeks are you training?', 1, 19)
        hoursPerWeek = st.number_input('How many hours are you training per week?', 1, 10)
        intensity = st.selectbox("**How intense do you want your training to be?**", ['High', 'Medium', 'Low'])
        numericIntensity = None
        if intensity == 'High':
            numericIntensity = 0
        elif intensity == 'Medium':
            numericIntensity = 1
        else:
            numericIntensity = 2    
        
        data = {'Training_hours_per_week': [hoursPerWeek], 
                'Starting_Weight_KG': [weight],
                'Duration_in_weeks': [duration],
                'Intensity': [numericIntensity]}
        df = pd.DataFrame(data)
        makePrediction = st.button('Make prediction')

        if makePrediction:
            prediction = randomForestModel.predict(df)
            predictionOutput = ""
            if prediction[0] > 0:
                match prediction:
                    case 0:
                        predictionOutput = '0 - 0.5kg'
                    case 1:
                        predictionOutput = '0.5 - 1.5kg'
                    case 2:
                        predictionOutput = '1.5 - 3.0kg'
                    case 3:
                        predictionOutput = '3.0 - 5.0kg'
                    case 4: 
                        predictionOutput = '5.0 - 7.0kg'
                    case 5:
                        predictionOutput = '7.0 - 10.0kg'
                    case 6:
                        predictionOutput = '10.0 - 14.0kg'
                    case 7:
                        predictionOutput = 'More than 14kg'
                st.success(f'With a starting weight of: {weight} kg, you will most likely have lost between {predictionOutput}, after having trained {duration} weeks, {hoursPerWeek} hours per week, at {intensity} intensity.')
            else:
                st.success("You won't lose any weight with what you have currently selected, try updating your training hours per week or the amount of weeks you're training.")
