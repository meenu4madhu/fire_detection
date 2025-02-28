import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# Load the trained model
model = tf.keras.models.load_model("C:\\Users\\Hp\\OneDrive\\Desktop\\model\\deployment\\fire_detection_model.h5")

# Function to preprocess image
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(299, 299))  # Resize for InceptionV3
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize
    return img_array

# Function to make predictions
def predict_fire(img_path):
    processed_img = preprocess_image(img_path)
    prediction = model.predict(processed_img)[0][0]  # Get prediction value
    print("Raw Prediction Score:", prediction)  # Debugging

    if prediction < 0.04:  # Ensure correct logic
        return "🔥 Fire Detected"
    else:
        return "✅ No Fire"

# Streamlit UI
st.title("🔥 Fire Detection System")

# Upload image section
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])
if uploaded_file:
    with open("uploaded_image.jpg", "wb") as f:
        f.write(uploaded_file.getbuffer())  # Save uploaded image

    st.image(uploaded_file, caption="Uploaded Image")
    result = predict_fire("uploaded_image.jpg")
    st.subheader(f"Prediction: {result}")

# Open camera and capture real-time visuals
if st.button("Open Camera"):
    cap = cv2.VideoCapture(0)  # Initialize camera
    if not cap.isOpened():
        st.error("Error: Could not access camera.")
    else:
        ret, frame = cap.read()
        if ret:
            cv2.imwrite("captured_frame.jpg", frame)  # Save captured image
            st.image(frame, channels="BGR", caption="Captured Image")
            
            result = predict_fire("captured_frame.jpg")
            st.subheader(f"Prediction: {result}")

        cap.release()  # Release camera properly
        cv2.destroyAllWindows()
