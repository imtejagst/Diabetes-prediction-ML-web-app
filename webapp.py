#description: diabetes detection using machinelearning in python
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from PIL import Image
import streamlit as st

#create a title and a subtitle
st.write("""
# Diabetes Detection
Detect if person has diabetes or not using machine learning
""")

#opening the image
image = Image.open('C:/Downloads/MLDLprojects/Diabetes detection web app using ML/diabetes detection image.png')
st.image(image, caption='Machine learning', use_column_width=True)

#getting the data
df = pd.read_csv('C:/Downloads/MLDLprojects/Diabetes detection web app using ML/diabetes.csv')

#setting a subheader
st.subheader('Data Information')

#set the data into table
st.dataframe(df)

#showing the statistics of the data
st.write(df.describe())

#showing the data of the chart
chart = st.bar_chart(df)

#splitting the data into independent variable 'X' and dependent variable 'Y'
X = df.iloc[:, 0:8].values
Y = df.iloc[:, -1].values

#splitting the data into 75 % training and 25 % testing 
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)

#getting the features from the user
def get_user_input():
    pregnancies = st.sidebar.slider('pregnancies', 0, 17, 3)
    glucose = st.sidebar.slider('glucose', 0, 199, 117)
    blood_pressure = st.sidebar.slider('blood_pressure', 0, 122, 72)
    skin_thickness = st.sidebar.slider('skin_thickness', 0, 99, 23)
    insulin = st.sidebar.slider('insulin', 0, 846, 30)
    BMI = st.sidebar.slider('BMI', 0, 67, 32)
    DPF = st.sidebar.slider('DPF', 0.078, 2.42, 0.3725)
    age = st.sidebar.slider('age', 21, 81, 29)

    #storing the dictionary into a variable
    user_data = {'pregnancies': pregnancies,
                 'glucose': glucose,
                 'blood_pressure': blood_pressure,
                 'skin_thickness': skin_thickness,
                 'insulin': insulin,
                 'BMI': BMI,
                 'DPF': DPF,
                 'age': age
                  }

    #transform the data into variable
    features = pd.DataFrame(user_data, index = [0])
    return features

#storing the input into a variable
user_input = get_user_input()

#set the subheader and display the user inputs
st.subheader('User input:')
st.write(user_input)

#creating the model using random forest
RandomForestClassifier = RandomForestClassifier()
RandomForestClassifier.fit(X_train, Y_train)

#showing the metrics
st.subheader('model test accuracy score:')
st.write( str(accuracy_score(Y_test,RandomForestClassifier.predict(X_test))*100)+'%')

#store the model predictions into a variable
prediction = RandomForestClassifier.predict(user_input)

#set a subheader for the display the clssification
st.subheader('Classification')
st.write(prediction)