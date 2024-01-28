import av
from collections import Counter
import mediapipe as mp
import cv2
import pandas as pd
import streamlit as st
from streamlit_webrtc import webrtc_streamer
import pickle
import numpy as np

mp_drawing = mp.solutions.drawing_utils # Draw the detections from the model to the screen
mp_holistic = mp.solutions.holistic # Mediapipe Solutions holistic model

with open(r"body_language_model_official_rfc2.pkl","rb") as f:
    model = pickle.load(f)

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

class Teacher:
    def __init__(self):
        self.data = []

    def append_data(self, data):
        self.data.append(data)
        generate_student_statistics()
        
    def generate_student_statistics(self):
        try:
            counter = Counter(self.data)
            pe_e = max(counter, key=counter.get)
            pe_prp = max(counter.values())/len(self.data)
            if pe_prp >= 0.5:
                st.write(f"{now}: Majority of your students are {pe_e}.")
                if pe_prp == "happy":
                    st.write(f"{now}: They are very engaged! Keep on teaching!")
                if pe_prp == "bored":
                    st.write(f"{now}: Try different engaging strategies, such as calling on students or fun quizzes.")
                if pe_prp == "confused":
                    st.write(f"{now}: Try explaining the concepts in a different way.")
                if pe_prp == "sad":
                    st.write(f"{now}: Check in on your students' mental and physical health!")
            st.bar_chart(data=counter, columns=['happy','bored','confused','sad'])
        except:
            st.write("The online session has not started yet!")

class Toggle:
    def __init__(self):
        self.on = st.toggle('Activate body language detection feature')
    def is_on(self):
        if self.on:
            return True
        else:
            return False

st.title("Body Language Detection for Online Learning")
st.subheader("This is the student interface.", divider='grey')
st.write("This is what will appear on a student's screen in full deployment.")

toggle = Toggle()
def callback(frame):
    img = frame.to_ndarray(format="bgr24")
    student = Student()
    teacher = Teacher()
    if toggle.is_on():  
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
            teacher.append_data(body_language_class.lower())
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
    return av.VideoFrame.from_ndarray(img, format="bgr24")


webrtc_streamer(key="example", video_frame_callback=callback, rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    })

st.subheader("This is the teacher interface.", divider='grey')
st.write("This is what will appear on a teacher's screen in full deployment.")
#generate_student_statistics(student_data)
