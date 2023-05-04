import streamlit as st
import numpy as np
from pickle import load

st.write("<h1 style='text-align: center; color: #FFD700;'>Penguin Species Predictor🐧</h1>", unsafe_allow_html=True)

st.image('penguin_logo.jpg')

scaler = load(open('models/standard_scaler.pkl', 'rb'))
knn_model = load(open('models/knn_model.pkl', 'rb'))
lr_model = load(open('models/lr_model.pkl', 'rb'))
dt_model = load(open('models/dt_model.pkl', 'rb'))
nb_model = load(open('models/nb_model.pkl', 'rb'))
rf_model = load(open('models/rf_model.pkl', 'rb'))
sv_model = load(open('models/sv_model.pkl', 'rb'))

col1, col2 = st.columns(2)
with col1:
    cl = st.text_input("Culmen Length", placeholder = "Enter  Length in mm")
with col2:
    cd = st.text_input("Culmen Depth", placeholder = "Enter  Depth in mm")

col3, col4 = st.columns(2)
with col3:
    fl = st.text_input("Flipper Length", placeholder = "Enter  Length in mm")
with col4:    
    bm = st.text_input("Body Mass", placeholder = "Enter  Mass in grams")

option = st.selectbox('Choose A Model:', ('K Nearest Neighbors', 'Logistic Regression'
                                        , 'Decision Tree', 'Naive Bayes', 'Random Forest', 'Support Vector Machine'))
st.write('You Selected:', option)

btn_click = st.button("Predict")

if option == 'K Nearest Neighbors':
    if btn_click == True:
        if cl and cd and fl and bm:
            query_point = np.array([float(cl), float(cd), float(fl), float(bm)])
            query_point = query_point.reshape(1, -1)
            query_point_transformed = scaler.transform(query_point)
            pred = knn_model.predict(query_point_transformed)
            st.success(pred[0])
        else:
            st.error("Enter The Values Properly.")

elif option == 'Logistic Regression':
    if btn_click == True:
        if cl and cd and fl and bm:
            query_point = np.array([float(cl), float(cd), float(fl), float(bm)])
            query_point = query_point.reshape(1, -1)
            query_point_transformed = scaler.transform(query_point)
            pred = lr_model.predict(query_point_transformed)
            st.success(pred[0])
        else:
            st.error("Enter The Values Properly.")

elif option == 'Decision Tree':
    if btn_click == True:
        if cl and cd and fl and bm:
            query_point = np.array([float(cl), float(cd), float(fl), float(bm)])
            query_point = query_point.reshape(1, -1)
            query_point_transformed = scaler.transform(query_point)
            pred = dt_model.predict(query_point_transformed)
            st.success(pred[0])
        else:
            st.error("Enter The Values Properly.")

elif option == 'Naive Bayes':
    if btn_click == True:
        if cl and cd and fl and bm:
            query_point = np.array([float(cl), float(cd), float(fl), float(bm)])
            query_point = query_point.reshape(1, -1)
            query_point_transformed = scaler.transform(query_point)
            pred = nb_model.predict(query_point_transformed)
            st.success(pred[0])
        else:
            st.error("Enter The Values Properly.")

elif option == 'Random Forest':
    if btn_click == True:
        if cl and cd and fl and bm:
            query_point = np.array([float(cl), float(cd), float(fl), float(bm)])
            query_point = query_point.reshape(1, -1)
            query_point_transformed = scaler.transform(query_point)
            pred = rf_model.predict(query_point_transformed)
            st.success(pred[0])
        else:
            st.error("Enter The Values Properly.")

elif option == 'Support Vector Machine':
    if btn_click == True:
        if cl and cd and fl and bm:
            query_point = np.array([float(cl), float(cd), float(fl), float(bm)])
            query_point = query_point.reshape(1, -1)
            query_point_transformed = scaler.transform(query_point)
            pred = sv_model.predict(query_point_transformed)
            st.success(pred[0])
        else:
            st.error("Enter The Values Properly.")

st.markdown("<hr>", unsafe_allow_html=True)
st.write("Note: All Models Have Similar Accuracy Hence Outputs Are Similar💯")