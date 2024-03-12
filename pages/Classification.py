import Code.readData as read


import streamlit as st
import pandas as pd
from PIL import Image


ut = read.loadData('Data/weight_loss_dataset.csv', 'csv')

st.set_page_config(page_title="Classification", layout="wide")

st.title("Classification")
st.sidebar.header("Classification", divider='rainbow')

tab1, tab2 = st.tabs(['Random Forest Classifier', 'Naive Bayes'])

with tab1:
    st.image(Image.open('Documentation/Graphs/classification/ClassificationReportTestData.png'))
    st.image(Image.open('Documentation/Graphs/classification/ConfusionMatrixDecisionTree.png'))
    st.success('Best result - Accuracy score = 0.50 +-0.05')


with tab2:
    st.image(Image.open('Documentation/Graphs/classification/ConfusionMatrixNaiveBayes.png'))
    st.success('Best result - Accuracy score = 0.40 +-0.06')
