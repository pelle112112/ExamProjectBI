import streamlit as st
from Code.languageModel import getResponse

st.set_page_config(page_title="Regression", layout="wide")

st.title("Regression")
st.sidebar.header("Regression", divider='rainbow')

st.success('👇 Below you can look for exercises in our exercise catalog, just ask what you are looking for. 👇')

input = st.text_input('What are you looking for?')
responseButton = st.button('Get a response')
if responseButton:
    response, similarityScore = getResponse(input)
    st.success('Hopefully this exercise matches what you are looking for.')
    st.table(response)
    st.write(f'Similarity score: {similarityScore}')