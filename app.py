import streamlit as st
import pandas as pd
import keras
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler, LabelEncoder
from joblib import load
from statistics import mode

st.title("Dominant Emotion Classifier Demo")
st.header("By: Aditi, Angela, Caleb, Coco, Roshan")

le = LabelEncoder()
le.classes_ = load("dtle.joblib")
dtsvmscaler = load("scaler.joblib")
minmaxscaler= load("minmaxscaler.joblib")

nn = keras.models.load_model("neuralnetwork.keras")
nn.compile(optimizer="Adam", loss="categorical_crossentropy", metrics=["accuracy"])
svm = load("svm.joblib")
dt = load("dt.joblib")
log_reg = load("log_reg.joblib")
rf = load("rf.joblib")
knn_eu = load("knn_euclidean.joblib")
knn_man = load("knn_manhattan.joblib")
knn_cos = load("knn_cosine.joblib")


emotion_labels = {
    "0" : "Anger",
    "1" : "Anxiety",
    "2" : "Boredom",
    "3" : "Happiness",
    "4" : "Neutral",
    "5" : "Sadness"
}

@st.experimental_dialog("Model Predictions")
def prediction(nn, svm, dt, logreg, rf, knneu, knnman, knncos):
    nnstring = "Neural Network Prediction (96.11%): " + emotion_labels[str(nn[0])] + "\n"
    svmstring = "SVM Prediction (95.00%): " + str(svm[0]) + "\n"
    dtstring = "Decision Tree Prediction (80.27%): " + emotion_labels[str(dt[0]-1)] + "\n"
    logregstring = "Logistic Regression Prediction (55.65%): " + emotion_labels[str(logreg[0])] + "\n"
    rfstring = "Random Forrest Prediction (88.31%): " + emotion_labels[str(rf[0])] + "\n"
    knneustring = "kNN Euclidean Distance Prediction (98.00%): " + emotion_labels[str(knneu[0])] + "\n"
    knnmanstring = "kNN Manhattan Distance Prediction (97.09%): " + emotion_labels[str(knnman[0])] + "\n"
    knncosstring = "kNN Cosine Distance Prediction (97.09%): " + emotion_labels[str(knncos[0])] + "\n"

    common = "Common Prediction: **" + mode([emotion_labels[str(nn[0])],str(svm[0]),emotion_labels[str(dt[0]-1)],emotion_labels[str(logreg[0])],emotion_labels[str(rf[0])],emotion_labels[str(knneu[0])],emotion_labels[str(knnman[0])],emotion_labels[str(knncos[0])]]) + "**"
    st.write(common)
    st.text(logregstring+knneustring+knnmanstring+knncosstring+dtstring+rfstring+svmstring+nnstring)
    

with st.form("input"):
    age = st.slider("What is your age?", 0, 100, step=1)
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
        x = {'Age' : [age], 
         'Gender' : [gender], 
         'Plat' : [platform], 
         'Daily_Usage_Time (minutes)' : [min_per_day], 
         'Posts_Per_Day' : [posts_per_day], 
         'Likes_Received_Per_Day' : [likes_received_per_day],
         'Comments_Received_Per_Day' : [comments_per_day],
         'Messages_Sent_Per_Day' : [msg_per_day]}
        x = pd.DataFrame(data=x)
        svmx = dtsvmscaler.fit_transform(x)
        dtx = dtsvmscaler.fit_transform(x[['Daily_Usage_Time (minutes)', 'Posts_Per_Day', 'Likes_Received_Per_Day', 'Comments_Received_Per_Day', 'Messages_Sent_Per_Day']])
        roshanx = x.copy
        roshanx[['Age', 'Daily_Usage_Time (minutes)', 'Posts_Per_Day', 'Likes_Received_Per_Day', 'Comments_Received_Per_Day', 'Messages_Sent_Per_Day']] = minmaxscaler.fit_transform(
    roshanx[['Age', 'Daily_Usage_Time (minutes)', 'Posts_Per_Day', 'Likes_Received_Per_Day', 'Comments_Received_Per_Day', 'Messages_Sent_Per_Day']])

        nnpred = nn.predict(x)
        nnpred = np.argmax(nnpred, axis=1)
        svmpred = svm.predict(svmx)
        dtpred = dt.predict(dtx)
        log_reg_pred = log_reg.predict(roshanx)
        rf_pred = rf.predict(roshanx)
        knneu_pred = knn_eu.predict(roshanx)
        knnman_pred = knn_man.predict(roshanx)
        knncos_pred = knn_cos.predict(roshanx)
        prediction(nnpred, svmpred, dtpred, log_reg_pred, rf_pred, knneu_pred, knnman_pred, knncos_pred)

