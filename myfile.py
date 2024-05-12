import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont

# Function to load TensorFlow Lite model
@st.cache(allow_output_mutation=True)
def load_model(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    return interpreter, input_details, output_details

# Function to load labels from file
def load_labels(labels_path):
    with open(labels_path, 'r') as f:
        labels = f.readlines()
        labels = [label.strip() for label in labels]
    return labels

# Function to preprocess frame data and resize it
def preprocess_frame(frame, target_size=(224, 224)):
    resized_frame = tf.image.resize(frame, target_size)
    processed_frame = resized_frame.numpy().astype(np.float32) / 255.0
    return processed_frame

# Function to perform object detection
def detect_objects(interpreter, input_details, output_details, frame):
    interpreter.set_tensor(input_details[0]['index'], np.expand_dims(frame, axis=0))
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    return output

# Function to add text to the frame
def add_text_to_frame(frame, text):
    pil_image = Image.fromarray(frame)
    draw = ImageDraw.Draw(pil_image)
    font = ImageFont.truetype("arial.ttf", 36)
    draw.text((10, 30), text, font=font, fill=(0, 255, 0))
    return np.array(pil_image)

# Load the TensorFlow Lite model
model_path = "models/mobilenet_v1_1.0_224.tflite"
interpreter, input_details, output_details = load_model(model_path)

# Load labels from file
labels_path = "models/mobilenet_v1_1.0_224.txt"
labels = load_labels(labels_path)

# Define a video transformer class for object detection
class ObjectDetectionTransformer(VideoTransformerBase):
    def __init__(self):
        super().__init__()

    def transform(self, frame):
        processed_frame = preprocess_frame(frame)
        output = detect_objects(interpreter, input_details, output_details, processed_frame)
        detected_label = labels[np.argmax(output)]
        confidence = np.max(output)
        annotated_frame = add_text_to_frame(frame, f"{detected_label}   {confidence:.2f}")
        return annotated_frame

# Create the Streamlit app
st.title('Object Detection App')

# Display the webcam feed and perform object detection
webrtc_streamer(key="example", video_transformer_factory=ObjectDetectionTransformer)
