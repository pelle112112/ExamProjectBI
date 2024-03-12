import Code.readData as read


import streamlit as st
import pandas as pd
from PIL import Image


ut = read.loadData('Data/weight_loss_dataset.csv', 'csv')

st.set_page_config(page_title="Unsupervised Training", layout="wide")

st.title("Unsupervised Training")
st.sidebar.header("Unsupervised Training", divider='rainbow')

tab1, tab2, tab3 = st.tabs(['K-Means', 'Hierarchical Clustering', 'difference'])

with tab1:    
    st.image(Image.open('Documentation/Graphs/K-Mean/Silhouette Analysis.png'))
    st.success('''First we need to determend how many clusters we need.
               The Silhouette Analysis shows that the score peaks at 9 clusters, but there isn't much difference between 5 and 9
               So we will use 5 clusters for the K-Means algorithm''')
    
    st.image(Image.open('Documentation/Graphs/K-Mean/ClusterMap.png'))
    st.success('Next we create a cluster map to visualize the clusters')

    st.image(Image.open('Documentation/Graphs/K-Mean/Silhouette plot.png'))
    st.success('The silhouette plot shows that the clusters are well defined')
    st.success('Silhouette score =  0.5744787941043337')
    
                
with tab2: 
    st.image(Image.open('Documentation/Graphs/Hierarchical Clustering/Dendogram.png'))
    st.success('''The first the we make, is a Dendrogram to visualize the cluster options''')

    st.success('We let the algorithm decide the number of clusters')
    st.success('Number of clusters: 15')

    st.image(Image.open('Documentation/Graphs/Hierarchical Clustering/ClusterMap.png'))
    st.success('Next we create a cluster map to visualize the clusters')

    st.image(Image.open('Documentation/Graphs/Hierarchical Clustering/3D-scatterMap.png'))
    st.success('The 3D scatter map shows that the clusters are well defined')


    st.success('Best result - Silhouette score: 0.7039061242185103')

   
    
with tab3:
    st.success ('K-Means and Hierarchical Clustering are both unsupervised learning algorithms, but they are very different in how they work')
    st.success('''The K-Means Clustering is simple to read, and works well on small and large datasets
               But you need to know the number of clusters beforehand''')
    
    st.success('''The Hierarchical Clustering is more complex to read, and works well on small datasets.
               But you don't need to know the number of clusters beforehand.
               Problem is that it is not optimal to use on larger datasets, as it is very workload heavy.''')