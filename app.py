import streamlit as st
import pandas as pd
import keras
import numpy as np
from joblib import load

st.title("Dominant Emotion Classifier Demo")
st.header("By: Aditi, Angela, Caleb, Coco, Roshan")

nn = keras.models.load_model("neuralnetwork.keras")
nn.compile(optimizer="Adam", loss="categorical_crossentropy", metrics=["accuracy"])

emotion_labels = {
    '0' : 'Anger',
    '1' : 'Anxiety',
    '2' : 'Boredom',
    '3' : 'Happiness',
    '4' : 'Neutral',
    '5' : 'Sadness'
}

with st.form("input"):
    age = st.number_input("What is your age?", 0, 100, 0, 1)
    gender = st.selectbox("What is your gender?", ["Male", "Female", "Non-binary"])
    platform = st.selectbox("Which social media platform do you use?", ["Facebook", "Instagram", "LinkedIn", "Snapchat", "Telegram", "Twitter", "Whatsapp"])
    min_per_day = st.number_input("How many minutes a day do you use this platform?")
    posts_per_day = st.number_input("How many posts do you post per day?")
    likes_received_per_day = st.number_input("How many likes do you receive per day?")
    comments_per_day = st.number_input("How many comments do you receive per day?")
    msg_per_day = st.number_input("How many messages do you send per day? (Includes sending vids/imgs)")

    submitted = st.form_submit_button("Submit")
    if gender == "Female":
        gender = 0
    elif gender == "Male":
        gender = 1
    elif gender == "Non-binary":
        gender = 2
    
    if platform == "Facebook":
        platform = 0
    elif platform == "Instagram":
        platform = 1
    elif platform == "LinkedIn":
        platform = 2
    elif platform == "Snapchat":
        platform = 3
    elif platform == "Telegram":
        platform = 4 
    elif platform == "Twitter":
        platform = 5
    elif platform == "Whatsapp":
        platform = 6

    if submitted:
        x = np.array([[age, gender, platform, min_per_day, posts_per_day, likes_received_per_day, comments_per_day, msg_per_day]])
        pred = nn.predict(x)
        pred = np.argamx(pred, axis=1)
        st.write("Our model predicts that your dominant emotion is: ", emotion_labels[pred])
