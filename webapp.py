import json
import time
import av
import matplotlib.pyplot as plt
import mediapipe as mp
import cv2
import pandas as pd
import requests
import streamlit as st
from streamlit_webrtc import webrtc_streamer
import pickle
import numpy as np
import threading
from collections import Counter
from datetime import datetime
from gtts import gTTS
import os

mp_drawing = mp.solutions.drawing_utils # Draw the detections from the model to the screen
mp_holistic = mp.solutions.holistic # Mediapipe Solutions holistic model

finished = True
lock = threading.Lock()
job_applicant_container = {'happy': 1, 'bored': 1, "confused": 1, 'sad': 1}

st.set_page_config(
    page_title="Job Interview Simulator",
    layout="wide"
)

st.title("Job Interview Simulator")
st.write("This tool will track your body language when responding to AI interviewer. This real-time feedback will train for the real thing.")
st.image(use_column_width="always",image="pexels-thisisengineering-3861969.jpg")

def generate_advice_for_applicant(occupation, n, majority_emotion=""):
    if n == 1:
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            data=json.dumps({
                "models": ["nousresearch/nous-capybara-7b", "mistralai/mistral-7b-instruct","huggingfaceh4/zephyr-7b-beta"],
                "messages": [{"role": "user", "content": f"You are applying to be a {occupation}. Having just completed your job interview, the body language most commonly detected was {majority_emotion}. Based on this, assuming the POV of a job interviewer, generate advice for how to improve body language during an interview to maximize chances of being hired."}],
                "route": 'fallback'
            }),
            headers={"Authorization": f"Bearer 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"}
        )
    if n == 2:
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            data=json.dumps({
                "models": ["nousresearch/nous-capybara-7b", "mistralai/mistral-7b-instruct","huggingfaceh4/zephyr-7b-beta"],
                "messages": [{"role": "user", "content": f"This job applicant is applying to be a {occupation}. Based on this, assuming the POV of a job interviewer, generate 3 questions they would ask. If and only if it is a technical role, please include technical questions (ex. software engineer, program something in python using conditionals) and if it is not, do not give suggestions for technical questions."}],
                "route": 'fallback'
            }),
            headers={"Authorization": f"Bearer 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"}
        )
    r_json = response.json()
    content = r_json['choices'][0]['message']['content'].strip()
    return content

class Applicant:
    def __init__(self):
        self.detected_emotion = None

    def update_emotion(self, a_class):
        self.detected_emotion = a_class
        return self.send_message_to_applicant()

    def get_emotion(self):
        return self.detected_emotion

    def send_message_to_applicant(self):
        if self.detected_emotion == "happy":
            return "It is good that you are appear excited and eager--interviewers want to see interest!"
        if self.detected_emotion == "bored":
            return "Liven up! Even if this interview may be boring, or this job is not exactly what you want, you must appear engaged!"
        if self.detected_emotion == "confused":
            return "Don't forget you are evaluating this company's fit for you! If you need clarification, wait for a good stopping point, then ask! Asking shows interest and curiosity."
        if self.detected_emotion == 'sad':
            return "This process may be difficult, but chin up! Negative emotion could be misconstrued as disinterest or dissatisification with job details."

with open(r"body_language_model_official_gbc3.pkl","rb") as f:
    model = pickle.load(f)

st.session_state["msg"] = ""
def callback(frame):
    global job_applicant_container
    img = frame.to_ndarray(format="bgr24")
    new_applicant = Applicant()
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img.flags.writeable = False
        results = holistic.process(img)
        img.flags.writeable = True  # prevents copying the image data, we're able to use it for rendering
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        # pose
        pose = results.pose_landmarks.landmark
        pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())
        # face
        face = results.face_landmarks.landmark
        face_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in face]).flatten())
        # can eventually combine the hand landmarks too
        # combining them into one big array
        row = pose_row + face_row
        x = pd.DataFrame([row])
        body_language_class = model.predict(x)[0]  # first value of the predict array
        msg = new_applicant.update_emotion(body_language_class.lower())
        y_max = int(max([landmark.y for landmark in results.face_landmarks.landmark]) * 480)
        y_min = int(min([landmark.y for landmark in results.face_landmarks.landmark]) * 480)
        x_max = int(max([landmark.x for landmark in results.face_landmarks.landmark]) * 640)
        x_min = int(min([landmark.x for landmark in results.face_landmarks.landmark]) * 640)
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (255, 255, 255), 2)
        cv2.rectangle(img, (x_min, y_min - 30), (x_max + len(body_language_class), y_min), (245, 117, 16), -1)
        cv2.putText(img, body_language_class, (x_min + 2, y_min - 4), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                    cv2.LINE_AA)
    with lock:
        job_applicant_container[body_language_class.lower()] += 1
        st.session_state["msg"] = msg
    return av.VideoFrame.from_ndarray(img, format="bgr24")

st.header("Step 1: Pick Your Interviewer")
c3, c4 = st.columns(2)

def make_toggle_false():
    if st.session_state.t1:
        if st.session_state.t2:
             st.session_state.t2 = False
    elif st.session_state.t2:
        if st.session_state.t1:
            st.session_state.t1 = False
with c3:
    st.image("blackmaninterviewer1.jpg")
    st.checkbox("Black Male Interviewer", key="t1", on_change=make_toggle_false)
with c4:
    st.image("hispanicwomeninterviewer1.jpg")
    st.checkbox("Hispanic Female Interviewer", key="t2", on_change=make_toggle_false)

st.header("Step 2: Fill out the Form")
st.write("This is so we can generate a list of questions specific to the job you are applying to.")
with st.form("applicant_qna"):
    st.write("Please fill out the requested fields.")
    job = st.text_input("What job are you applying for? ex. software engineer")
    submitted = st.form_submit_button("Submit")
if submitted:
    st.session_state["job_applicant_qs"] = generate_advice_for_applicant(job,2)
        
st.header("Step 3: Start Your Mock Interview")
st.write("Answer the following questions while looking into the camera.")
c1, c2, c3 = st.columns(3)
with c1:
    ctx = webrtc_streamer(key="example", video_frame_callback=callback, rtc_configuration={
                    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
with c2:
    if st.session_state.t1:
        st.image("blackmaninterviewer1.jpg")
    if st.session_state.t2:
        st.image("hispanicwomeninterviewer1.jpg")
with c3:
    st.write("Questions loading...")

try:        
    stutter = st.empty()
    while ctx.state.playing:
        with lock:
            finished = False
        with stutter.container():
            with lock:
                data = job_applicant_container
                labels = list(job_applicant_container.keys())
                counts = list(job_applicant_container.values())
            with c3:
                st.write(st.session_state["job_applicant_qs"])
    
            st.write(st.session_state["msg"])
            pe_e = max(data, key=data.get)
            st.write(f"Your body language mostly indicates you are {pe_e}.")
            fig, ax = plt.subplots()
            ax.pie(counts, labels=labels, autopct='%.2f')
            ax.axis('equal')
            st.pyplot(fig)
            with open("dump.txt", 'w') as file:
                file.write(pe_e)
            time.sleep(5)
except:
    pass
        
with open("dump.txt") as file:
    main_emotion = file.read()

# program started and captured new data
if finished:
    result = st.subheader("Get Feedback from Chatbot")
    with st.form("applicant_feedback"):
        st.write("Please fill out the requested fields.")
        job = st.text_input("What job were you applying for? ex. software engineer")
        submitted = st.form_submit_button("Submit")
    if submitted:
        cnt = generate_advice_for_applicant(job, 1, main_emotion)
        st.write(cnt)
