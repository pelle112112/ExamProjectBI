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
    st.image(Image.open('Documentation/Graphs/classification/ClassificationReportTestDataUPD.png'))
    st.image(Image.open('Documentation/Graphs/classification/ClassificationReportTrainingDataUPD.png'))
    st.success('The training data has the best f1-score according to the classification reports, which we expected.')
    st.success('After adding bins (intervals) to the weight loss, we got a better result, as the model now has a better understanding of the data.')
    st.success('We didnt expect the random forest classifier to be a good model for predicting weight loss, but it turned out to be one of the best models.')
    st.image(Image.open('Documentation/Graphs/classification/ConfusionMatrixDecisionTree.png'))
    st.success('The confusion matrix for the random forest classifier tells us that we our model looks good, with almost no FP and FN.')
    st.image(Image.open('Documentation/Graphs/classification/FeatureImportanceClassificationTree.png'))
    st.success('The feature importance of the random forest classifier tells us that the most important feature is the training hours per week, which we expected.')
    st.image(Image.open('Documentation/Graphs/classification/RandomForestClassifierModel.png'))
    st.success('Best result - Accuracy score = 0.79 +-0.06')


with tab2:
    st.image(Image.open('Documentation/Graphs/classification/ConfusionMatrixNaiveBayes.png'))
    st.success('The Naive Bayes models Confusion Matrix mostly looks the same as the random forest classifier, but with more false positives.')
    st.success('Best result - Accuracy score = 0.62 +-0.06')
    st.success('Due to the high accuracy score of the random forest Classifer, we did not include the Naive Bayes model in the final product.')
