import streamlit as st
import pandas as pd
import keras
from joblib import load

st.title("Dominant Emotion Classifier Demo")
st.header("By: Aditi, Angela, Caleb, Coco, Roshan")

nn = keras.models.load_model("neuralnetwork.keras")

def getUserInput():
    age = st.number_input("What is your age?", 0, 100, 0, int)
    gender = st.selectbox("What is your gender?", ["Male", "Female", "Non-binary"])
    platform = st.selectbox("Which social media platform do you use?", ["Facebook", "Instagram", "LinkedIn", "Snapchat", "Telegram", "Twitter", "Whatsapp"])
    min_per_day = st.number_input("How many minutes a day do you use this platform?")
    posts_per_day = st.number_input("How many posts do you post per day?")
    likes_received_per_day = st.number_input("How many likes do you receive per day?")
    comments_per_day = st.number_input("How many comments do you receive per day?")
    msg_per_day = st.number_input("How many messages do you send per day? (Includes sending vids/imgs)")
    return [[age], [gender], [platform], [min_per_day], [posts_per_day], [likes_received_per_day], [comments_per_day], [msg_per_day]]

x = getUserInput()

st.write(nn.predict(x))