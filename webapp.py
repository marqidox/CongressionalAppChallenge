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
mp_drawing = mp.solutions.drawing_utils # Draw the detections from the model to the screen
mp_holistic = mp.solutions.holistic # Mediapipe Solutions holistic model

finished = True
lock = threading.Lock()
student_emotion_container = {'happy': 1, 'bored': 1, "confused": 1, 'sad': 1}
class Student:
    def __init__(self):
        self.detected_emotion = None

    def update_emotion(self, s_class):
        self.detected_emotion = s_class
        return self.send_message_to_student()

    def get_emotion(self):
        return self.detected_emotion

    def send_message_to_student(self): # works
        # could make a bank of messages and select randomly from it
        if self.detected_emotion == "happy":
            return "Give a thumbs up in the chat!"
        if self.detected_emotion == "bored":
            return "Stay engaged by being active! Or, ask for a break."
        if self.detected_emotion == "confused":
            return "Raise your hand to ask the teacher a question."
        if self.detected_emotion == 'sad':
            return "Talk to someone to process your emotions."

st.title("Body Language Detection for Online Learning")
st.header("Student Interface")
#st.toggle("Activate Body Language Detector")
with open(r"body_language_model_official_rfc3.pkl","rb") as f:
    model = pickle.load(f)

def callback(frame):
    global student_emotion_container
    img = frame.to_ndarray(format="bgr24")
    new_student = Student()
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
        msg = new_student.update_emotion(body_language_class.lower())
        y_max = int(max([landmark.y for landmark in results.face_landmarks.landmark]) * 480)
        y_min = int(min([landmark.y for landmark in results.face_landmarks.landmark]) * 480)
        x_max = int(max([landmark.x for landmark in results.face_landmarks.landmark]) * 640)
        x_min = int(min([landmark.x for landmark in results.face_landmarks.landmark]) * 640)
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (255, 255, 255), 2)
        cv2.rectangle(img, (x_min, y_min - 30), (x_max + len(body_language_class), y_min), (245, 117, 16), -1)
        cv2.putText(img, body_language_class, (x_min + 2, y_min - 4), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                    cv2.LINE_AA)
        cv2.rectangle(img, (0, 0), (0 + len(msg) * 15 + 7, 35), (0, 0, 0), -1)
        cv2.rectangle(img, (2, 2), (0 + len(msg) * 15 + 5, 33), (255, 255, 255), -1)
        cv2.putText(img, msg, (9, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    with lock:
        student_emotion_container[body_language_class.lower()] += 1
    return av.VideoFrame.from_ndarray(img, format="bgr24")


ctx = webrtc_streamer(key="example", video_frame_callback=callback, rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    })
st.header("This is the teacher interface.")
st.write("This is what will appear on a teacher's screen in full deployment.")
stutter = st.empty()
while ctx.state.playing:
    with lock:
        finished = False
    with stutter.container():
        now = datetime.now().astimezone().strftime("%I:%M%p (%Z)")
        st.write("The time is: " + str(now))
        with lock:
            data = student_emotion_container
            labels = list(student_emotion_container.keys())
            counts = list(student_emotion_container.values())
        pe_e = max(data, key=data.get)
        st.write(f"Majority of your students are {pe_e}.")
        if pe_e == "happy":
            st.write("They are very engaged! Keep on teaching!")
        if pe_e == "bored":
            st.write("Try different engaging strategies, such as calling on students or fun quizzes.")
        if pe_e == "confused":
            st.write("Try explaining the concepts in a different way.")
        if pe_e == "sad":
            st.write("Check in on your students' mental and physical health!")
        fig, ax = plt.subplots()
        ax.pie(counts, labels=labels, autopct='%.2f')
        ax.axis('equal')
        st.pyplot(fig)
        with open("dump.txt", 'w') as file:
            file.write(pe_e)
        time.sleep(5)

with open("dump.txt") as file:
    main_emotion = file.read()
def generate_report_for_teacher(majority_emotion, subject, grade):
    response = requests.post(
        url="https://openrouter.ai/api/v1/chat/completions",
        data=json.dumps({
            "models": ["nousresearch/nous-capybara-7b", "mistralai/mistral-7b-instruct","huggingfaceh4/zephyr-7b-beta"],
            "messages": [{"role": "user", "content": f"You are teaching the subject {subject} to students in {grade} online. Given that the emotion most commonly detected from your students was {majority_emotion}, generate advice for better instruction and teaching."}],
            "route": 'fallback'
        }),
        headers={"Authorization": f"Bearer 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"}
    )
    r_json = response.json()
    content = r_json['choices'][0]['message']['content'].strip()
    return content

# program started and captured new data
if finished:
    st.write(f'Overall, most of your students were {main_emotion}.')
    result = st.subheader("Get Feedback from Chatbot")
    with st.form("teacher_feedback"):
        st.write("Please fill out the requested fields.")
        subject = st.text_input("What subject do you teach?")
        grade = st.selectbox(
            "What grade level do you teach?",
            ("elementary", "middle school", 'high school', 'undergraduate college', 'graduate school')
        )
        submitted = st.form_submit_button("Submit")
    if submitted:
        cnt = generate_report_for_teacher(main_emotion, subject, grade)
        st.write(cnt)

