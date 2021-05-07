#%%
#Imports
import pandas as pd
import chime
import streamlit as st

## Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import sklearn as skl
import time
from IPython.display import display, clear_output
#from ctgan import CTGANSynthesizer
import sklearn
from sklearn import pipeline      # Pipeline
from sklearn import preprocessing # OrdinalEncoder, LabelEncoder
from sklearn import model_selection # train_test_split
from sklearn import metrics         # accuracy_score, balanced_accuracy_score, plot_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier

#%%
#Load data
original_data = pd.read_csv("C:\Users\gabri\Documents\Github\hippocratia\datasets")
#%%
#Preprocess data
#%% md

## boxplot from blood pressure (trestbps)

#%%
stats = original_data.trestbps.describe()
#low range = below 25% normal = 25% < 75%  high = above 75%
var_min, var_max, var_25, var_75 = stats[3], stats[7], stats[4], stats[6]

#%% md
## encoding blood pressure
#%%
#creating new column for classifying blood pressure:
# 0 = low, 1 = normal, 2 = high
df = original_data
for row in df.trestbps:
    if row < var_25:
        df['trestbps_encoded'] = 0
    elif var_25 < row < var_75:
        df['trestbps_encoded'] = 1
    elif var_75 < row:
        df['trestbps_encoded'] = 2

#Adding default values
age = 20
#Inputs
age = st.text(20)
st.text_input('Age')
sex = st.text_input('Sex', 'type here:')
cp = st.text_input('Chest Pain', 'type here:')
trestbps = st.text_input('Blood pressure', 'type here:')
chol = st.text_input('Cholesterine level', 'type here:')
fbs = st.text_input('Fasting blood sugar', 'type here:')
restecg = st.text_input('Resting Electrokardiographical result', 'type here:')
thalach = st.text_input('Maximum heart rate', 'type here:')
exang = st.text_input('Exercise includes Angina', 'type here:')
oldpeak = st.text_input('Oldpeak', 'type here:')
slope = st.text_input('Slope', 'type here:')
ca = st.text_input('Number of major vessels', 'type here:')
thal = st.text_input('Thal')

#Model
x = df.drop(columns="target", axis=1)
y = df[["target"]]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=40, stratify=y)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
model = KNeighborsClassifier()
model.fit(x_train, y_train)
score = model.score(x_test, y_test)
print(score)

#Predict
my_y = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
predict = model.predict(my_y)
st.dataframe(predict)
#predict = model.predict(my_y)
#pred_proba = model.predict_proba(my_y)
print(type(y_test))
#Display prediction with sound
audio_file = open('alarm.mp3', 'rb').read()

if predict == 1:
    chime.warning()
    st.audio(audio_file, format='audio/mp3')
else:
    print('You are good', predict)