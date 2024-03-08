import streamlit as st
import Code.readData as read
from PIL import Image

st.set_page_config(page_title="Data Exploration")

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
        tab1, tab2, tab3, tab4 = st.tabs(['Nothing', 'to', 'show', 'yet'])
        with tab1:
            st.write('put stuff here')
        with tab2:
            st.write('put stuff here')
        with tab3:
            st.write('put stuff here')
        with tab4:
            st.write('put stuff here')