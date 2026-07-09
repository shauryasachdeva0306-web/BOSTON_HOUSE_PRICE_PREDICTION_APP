import streamlit as st  # api integration
import pandas as pd     # data cleaning and preproceessing
import numpy as np      # mathematical computations
import pickle           #  load model file

st.set_page_config(
    page_title = "AI Powered House Price Prediction",
    page_icon = '🏘️',
    layout = 'wide')

st.markdown("""
<style>

.main{background:#f5f8ff}

.stButton>button{
background:#0099ff;
color:white;
border-radius:12px;
height:50px;
width:100%;
font-size:18px;}

h1{color:#003366}

.metric{font-size: 30px;}

<style>
""",unsafe_allow_html=True)


st.title("Boston House Price Prediction Powered by AI")
st.write("Here we are doing prediction of ouse prices using Machine Learning")

model= pickle.load(open('model.pkl','rb'))
columns= pickle.load(open('columns.pkl','rb'))

data= {}
col1,col2,col3 = st.columns(3)

for i,col in enumerate(columns):
    if i%3==0:
        with col1:
            data[col]=st.number_input(col,value=0.0)
    elif i%3==1:
        with col2:
            data[col]=st.number_input(col,value=0.0)
    else:
        with col3:
            data[col]=st.number_input(col,value=0.0)

if st.button("Prediction House Price"):
    df= pd.DataFrame([data])
    prediction = model.predict(df)[0]
    st.success(f' Estimated house price is ${prediction}')
    st.balloons()
st.divider
st.subheader("Project Highlights")

st.markdown("""
Linear Regression
Feature Engineering
Data Cleaning
Outlier Detection
StandardScaler
MinMaxScaler
Ridge
Lasso
ElasticNet
Adjusted R2
Deployment """)