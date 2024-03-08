import streamlit as st
import pandas as pd
import Code.readData as read
from PIL import Image

tab = read.loadData('Data/weight_loss_dataset.csv', 'csv')

st.set_page_config(page_title="Regression", layout="wide")

st.title("Regression")
st.sidebar.header("Regression", divider='rainbow')


tab1, tab2, tab3, tab4 = st.tabs(['Exploration of highest correlation', 'Linear Regression', 'Multilinear Regression', "Polynomial Regression"])
with tab1:    
    st.image(Image.open('Documentation/Graphs/regression/WeightLossDataScatterplotHoursPerWeekWeightLoss.png'))
    st.success('Data looks suitable for linear regression')
                
with tab2: 
    st.image(Image.open('Documentation/Graphs/regression/WeightLossDataLinearRegression.png'))
    st.success('Best result - RMSE score = 1.869 - R-squared score = 0.482')
      
with tab3:
    data = {'Features': ['Training_hours_per_week, Intensity, Duration_in_weeks, Starting_Weight_KG',
                        'Training_hours_per_week, Duration_in_weeks, Starting_Weight_KG',
                        'Training_hours_per_week, Intensity, Starting_Weight_KG', 
                        'Intensity, Duration_in_weeks, Starting_Weight_KG',
                        'Training_hours_per_week, Duration_in_weeks',
                        'Intensity, Starting_Weight_KG',
                        'Training_hours_per_week, Starting_Weight_KG',
                        'Duration_in_weeks, Starting_Weight_KG'], 
            'AIC score': [5095.726, 5308.397, 6035.168, 6527.301, 5546.852, 6958.684, 6130.616, 6618.216]}
    df = pd.DataFrame(data)
    st.dataframe(df)
    st.success('Best result - RMSE score = 1.278 - R-squared score = 0.758')

with tab4: 
    st.image(Image.open('Documentation/Graphs/regression/Polyregression.png'))
    st.success('Best result - RMSE score = 1.874 - R-squared score = 0.480')