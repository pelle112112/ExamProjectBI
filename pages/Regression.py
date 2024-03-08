import streamlit as st
import Code.readData as read

tab = read.loadData('../Data/weight_loss_dataset.csv')

st.set_page_config(page_title="Regression")

st.title("Regression")
st.sidebar.header("Regression", divider='rainbow')