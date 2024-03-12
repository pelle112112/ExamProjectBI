import streamlit as st
import Code.readData as readData
import Code.recommenderSystem as recommender
import pandas as pd

st.set_page_config(page_title="Recommender System", layout="wide")

st.title("Recommender System")
st.sidebar.header("Recommender System", divider='rainbow')

data = readData.loadData('Data/exercise_dataset.csv', 'csv')

feature = 'Activity, Exercise or Sport (1 hour)'

exercises = {}

for index, row in enumerate(data[feature]):
    activityType = row.split(',')[0]
    if activityType not in exercises:
        exercises[activityType] = [row]
    else:
        exercises[activityType].append(row)

st.success('👇 Choose an exercise you like below to find exercises you might like 👇')

exerciseType = st.selectbox('Exercise type:', exercises.keys())

chosenExercise = st.selectbox('Choose an exercise:', exercises[exerciseType]) 

recommenderButton = st.button('Get Recommendation')

if recommenderButton:
    input = data.loc[data[feature] == chosenExercise]
    input = input[feature].iloc[0] + ' ' + input['Calories per kg'].iloc[0].astype(str)
    recommendation = recommender.recommend(input, data)
    st.success(f'You might like the following exercises: ')
    output = pd.DataFrame(data={'Exercises':recommendation})
    output.index += 1
    st.table(output)