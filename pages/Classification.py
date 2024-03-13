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
    st.success('The training data has the best f1-score according to the classification reports, which we expected, but the test data has a worse score than anticipated.')
    st.success('We can conclude that we have too many classes, and that the model is not optimal for this dataset, as the accuracy score is only 0.59 +-0.05')
    st.success('We however didnt expect the random forest classifier to be a good model anyway for predicting weight loss, as it defines classes and not numerical values.')
    st.image(Image.open('Documentation/Graphs/classification/ConfusionMatrixDecisionTree.png'))
    st.success('The confusion matrix for the random forest classifier tells us that we have a few false negatives, but mostly it looks okay.')
    st.image(Image.open('Documentation/Graphs/classification/FeatureImportanceClassificationTree.png'))
    st.success('The feature importance of the random forest classifier tells us that the most important feature is the training hours per week, which we expected.')
    st.image(Image.open('Documentation/Graphs/classification/RandomForestClassifierModel.png'))
    st.success('Best result - Accuracy score = 0.59 +-0.05')


with tab2:
    st.image(Image.open('Documentation/Graphs/classification/ConfusionMatrixNaiveBayes.png'))
    st.success('The Naive Bayes models Confusion Matrix mostly looks the same as the random forest classifier, but with some false positives instead of false negatives.')
    st.success('Best result - Accuracy score = 0.40 +-0.06')
    st.success('Due to the low accuracy score we didnt want to include the naive bayes model in the final product, as it would not be optimal for the user.')
