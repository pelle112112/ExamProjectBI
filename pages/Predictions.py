import joblib
import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder as le

linearModel = joblib.load('model/bestLinearModel.pkl')

st.set_page_config(page_title="Predictions", layout="wide")

st.title("Predictions")
st.sidebar.header("Predictions", divider='rainbow')

chosenModel = st.selectbox('**Choose the model you want to make a prediction with**', ['Linear Regression'])

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
            st.success(f'With a starting weight of: {weight} kg, you will most likely have lost {prediction[0][0]} kg, after having trained {duration} weeks, {hoursPerWeek} hours per week, at {intensity} intensity.')