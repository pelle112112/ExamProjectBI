import streamlit as st
import Code.readData as read
from PIL import Image

st.set_page_config(page_title="Data Exploration", layout="wide")

st.title("Data Exploration")
st.sidebar.header("Data Exploration", divider='rainbow')

chosenExplorationData = st.selectbox('**Choose which dataset you want to explore**', ['Weight loss data', 'Exercise data'])

match chosenExplorationData:
    case 'Weight loss data':
        data = read.loadData('Data/weight_loss_dataset.csv', 'csv')
        st.dataframe(data)
        tab1, tab2, tab3 = st.tabs(['Histogram', 'Weight loss per intensity', "Correlation heatmap"])
        with tab1:    
            st.image(Image.open('Documentation/Graphs/exploration/weightlossDataHistogram.png'))
                
        with tab2: 
            st.image(Image.open('Documentation/Graphs/exploration/weightLossDataWeightLostPerIntensity.png'))
      
        with tab3: 
            st.image(Image.open('Documentation/Graphs/exploration/weightLossDataHeatmap.png'))
    case 'Exercise data':
        data = read.loadData('Data/exercise_dataset.csv', 'csv')
        st.dataframe(data)
        tab1, tab2, tab3, tab4 = st.tabs(['Gym Exercises', 'Starting Weight Boxplots', 'Histogram', 'Correlation heatmap'])
        with tab1:
            st.image(Image.open('Documentation/Graphs/exploration/Exercises_Users_Can_Do_At_Gym_Updated.png'))
            st.write('The exercises that users can do at the gym, and how many calories they burn if done for 1 hour')
            
        with tab2:
            st.image(Image.open('Documentation/Graphs/exploration/BoxPlot_Exercise_Data.png'))
            st.write('Correlation between starting weight and weight loss')
            st.image(Image.open('Documentation/Graphs/exploration/BoxPlot_Exercise_Data_Standardized.png'))
            st.write('Standaridized boxplot of the same data')
        with tab3:
            st.image(Image.open('Documentation/Graphs/exploration/All_Exercises_Calories.png'))
            st.write('Histogram of all exercises and the amount of calories they burn')
        with tab4:
            st.image(Image.open('Documentation/Graphs/exploration/Correlation_All_Exercises.png'))