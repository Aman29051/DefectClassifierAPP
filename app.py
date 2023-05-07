import streamlit as st
from PIL import Image
import cv2
import pickle
import io
import numpy as np
from keras.models import load_model
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer

# Set page title, icon and layout
st.set_page_config(page_title="Defect Classifier",page_icon="memo",layout="wide")

# Open header image file
img1 = Image.open('header.png')

# Display header and sidebar
st.header("Defect Classifier Model")
st.sidebar.image(img1)
st.sidebar.header("Predict Defect")

# Define a VideoTransformer class to capture frames from the camera
class VideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        # Convert the frame to RGB color space
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame

# Function to capture image from camera
def get_live_image():
    # Define the WebRTC video streamer
    webrtc_ctx = webrtc_streamer(key="example", video_processor_factory=LiveImageTransformer)

    # Check if the video streamer has started
    if not webrtc_ctx.video_transformer:
        st.warning("Error: Could not start video stream.")
        return None

    # Wait for the user to capture an image
    st.warning("Click the 'Capture' button to take a picture.")
    if st.button("Capture"):
        # Capture the current frame from the video streamer
        frame = webrtc_ctx.video_transformer.last_frame
        return frame

# Prompt user to select an option
option = st.sidebar.radio("Select an option", ("Upload Image", "Take Live Image"))

# If "Upload Image" is selected
if option == "Upload Image":
    # Prompt user to upload image
    inp_image = st.sidebar.file_uploader("Choose Input Image",type=["jpg","png","jpeg","bmp"])

    # Check if image is uploaded
    if inp_image is not None:
        st.image(inp_image, caption="Uploaded Image", use_column_width=True)
        # Read file contents into memory
        image_bytes = inp_image.read()

        # Create PIL Image object
        image_ = Image.open(io.BytesIO(image_bytes))
    
        # Load model and label binarizer
        lb = pickle.load(open("lb.pkl","rb"))
        model = load_model("model.h5")
    
        # Create numpy array from file buffer
        image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), 1)
        image = cv2.resize(image,(64,64))
        image = np.array(image)
        image = np.reshape(image,(1,64,64,3))

        # Make predictions and display result
        y_pred = model.predict(image)
        y_predicted_labels = [np.argmax(i) for i in y_pred]
        labels = lb.classes_
        label = labels[y_predicted_labels[0]]
        if st.sidebar.button("Predict Defect"):
            st.subheader("Predicted Class : {}".format(label))

# If "Take Live Image" is selected
elif option == "Take Live Image":
    # Prompt user to capture image
    if st.sidebar.button("Capture and Predict"):
        # Capture image from camera
        live_image = get_live_image()

        # Check if image is captured
        if live_image is not None:
            st.image(live_image, caption="Live Image", use_column_width=True)
    
            # Load model and label binarizer
            lb = pickle.load(open("lb.pkl","rb"))
            model = load_model("model.h5")
    
            # Resize image and convert to numpy array
            image = cv2.resize(live_image, (64, 64))
            image = np.array(image)
            image = np.reshape(image, (1, 64, 64, 3))

            # Make predictions and display result
            y_pred = model.predict(image)
            y_predicted_labels = [np.argmax(i) for i in y_pred]
            labels = lb.classes_
            label = labels[y_predicted_labels[0]]
            st.subheader("Predicted Class : {}".format(label))
