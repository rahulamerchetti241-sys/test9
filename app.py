import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"

import streamlit as st
import cv2
import numpy as np
import pickle
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="AI Sign Language Translator")
st.title("AI Sign Language Translator (ASL)")

# ---------------- CONFIG ----------------
STABILITY_FRAMES = 20
CONFIDENCE_THRESHOLD = 0.85


# ---------------- SAFE LOADERS ----------------
@st.cache_resource
def load_tf_model():
    import tensorflow as tf   # ✅ LOCAL import
    return tf.keras.models.load_model("sign_language_model.keras")


@st.cache_resource
def load_label_encoder():
    with open("label_encoder.pkl", "rb") as f:
        return pickle.load(f)


@st.cache_resource
def load_mediapipe():
    import mediapipe as mp

    # 🔒 SAFETY CHECK
    if not hasattr(mp, "solutions"):
        raise RuntimeError(
            "Invalid mediapipe package installed. "
            "Make sure mediapipe==0.10.9 is used."
        )

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )

    return hands, mp_hands, mp_drawing


# ---------------- LOAD RESOURCES ----------------
try:
    model = load_tf_model()
    encoder = load_label_encoder()
    hands, mp_hands, mp_drawing = load_mediapipe()
except Exception as e:
    st.error(f"❌ Resource loading failed: {e}")
    st.stop()


# ---------------- VIDEO PROCESSOR ----------------
class SignRecognizer(VideoTransformerBase):
    def __init__(self):
        self.sentence = []
        self.last_pred = None
        self.frame_counter = 0

    def transform(self, frame):
        image = frame.to_ndarray(format="bgr24")
        image = cv2.flip(image, 1)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        results = hands.process(rgb)

        predicted_char = "Waiting..."
        hand_detected = False

        if results.multi_hand_landmarks:
            hand_detected = True

            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y])

                input_data = np.array([landmarks])
                prediction = model.predict(input_data, verbose=0)

                class_id = np.argmax(prediction)
                confidence = np.max(prediction)

                if confidence > CONFIDENCE_THRESHOLD:
                    predicted_char = encoder.inverse_transform([class_id])[0]

                    if predicted_char == self.last_pred:
                        self.frame_counter += 1
                    else:
                        self.frame_counter = 0
                        self.last_pred = predicted_char

                    if self.frame_counter >= STABILITY_FRAMES:
                        p = predicted_char.lower()
                        if "space" in p:
                            self.sentence.append(" ")
                        elif "delete" in p and self.sentence:
                            self.sentence.pop()
                        else:
                            self.sentence.append(predicted_char)

                        self.frame_counter = 0

        # ---------------- UI ----------------
        h, w, _ = image.shape

        overlay = image.copy()
        cv2.rectangle(overlay, (0, 0), (w, 90), (0, 0, 0), -1)
        image = cv2.addWeighted(overlay, 0.6, image, 0.4, 0)

        status_color = (0, 255, 0) if hand_detected else (0, 0, 255)
        cv2.circle(image, (w - 30, 30), 12, status_color, -1)

        cv2.putText(image, "DETECTING:", (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

        cv2.putText(image, predicted_char, (160, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        sentence = "".join(self.sentence)
        cv2.putText(image, sentence[-25:], (15, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        return image


# ---------------- START STREAM ----------------
st.markdown("### ▶️ Click Start to activate camera")

if st.button("Start Camera"):
    webrtc_streamer(
        key="sign-language",
        video_transformer_factory=SignRecognizer,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )