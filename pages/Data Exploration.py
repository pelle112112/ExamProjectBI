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
            st.success('Histogram of the weight loss data')
                
        with tab2: 
            st.image(Image.open('Documentation/Graphs/exploration/weightLossDataWeightLostPerIntensity.png'))
            st.success('Weight loss per intensity')
            st.success('We can see that the higher the intensity, the more weight is lost, which is expected.')
      
        with tab3: 
            st.image(Image.open('Documentation/Graphs/exploration/weightLossDataHeatmap.png'))
            st.success('Correlation heatmap of the weight loss data')
            st.success('We can tell that the biggest correlation for weight loss (Excluding end weight) is the duration of the training, and the amount of hours trained per week.')
    case 'Exercise data':
        data = read.loadData('Data/exercise_dataset.csv', 'csv')
        st.dataframe(data)
        tab1, tab2, tab3, tab4 = st.tabs(['Gym Exercises', 'Starting Weight Boxplots', 'Histogram', 'Correlation heatmap'])
        with tab1:
            st.image(Image.open('Documentation/Graphs/exploration/Exercises_Users_Can_Do_At_Gym_Updated.png'))
            st.success('The exercises that users can do at the gym, and how many calories they burn if done for 1 hour')
            st.success('These are the exercises and different intensities that we have based our other data values on.')
            
        with tab2:
            st.success('These boxplots show the correlation between starting weight and weight loss. And we can tell that the correlation looks to be linear.')
            st.success('This means that the more you weigh, the more calories you burn, and the more weight you lose')
            st.success('Theres only a few outliers, which we wont remove, since they are just exercises that are very productive for the users.')
            st.image(Image.open('Documentation/Graphs/exploration/BoxPlot_Exercise_Data.png'))
            st.success('Correlation between starting weight and weight loss')
            st.image(Image.open('Documentation/Graphs/exploration/BoxPlot_Exercise_Data_Standardized.png'))
            st.success('Standaridized boxplot of the same data')
        with tab3:
            st.image(Image.open('Documentation/Graphs/exploration/All_Exercises_Calories.png'))
            st.success('Histogram of all exercises and the amount of calories they burn')
        with tab4:
            st.image(Image.open('Documentation/Graphs/exploration/Correlation_All_Exercises.png'))
            st.success('Correlation heatmap of all exercises')
            st.success('We can again see that the correlation between the body weights and amount of calories burned are 1=1.')