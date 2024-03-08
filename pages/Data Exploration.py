import streamlit as st
import Code.readData as read
from PIL import Image

tab = read.loadData('Data/weight_loss_dataset.csv', 'csv')

st.set_page_config(page_title="Data Exploration")

st.title("Data Exploration")
st.sidebar.header("Data Exploration", divider='rainbow')


tab1, tab2, tab3 = st.tabs(['Histogram', 'Weight loss per intensity', "Correlation heatmap"])
with tab1:    
    st.image(Image.open('Documentation/Graphs/exploration/weightlossDataHistogram.png'))
                
with tab2: 
    st.image(Image.open('Documentation/Graphs/exploration/weightLossDataWeightLostPerIntensity.png'))
      
with tab3: 
    st.image(Image.open('Documentation/Graphs/exploration/weightLossDataHeatmap.png'))