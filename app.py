import streamlit as st
from PIL import Image
import cv2
import pickle
import io
import numpy as np
from keras.models import load_model
import streamlit_webrtc as webrtc

# Set page title, icon and layout
st.set_page_config(page_title="Defect Classifier",page_icon="memo",layout="wide")

# Open header image file
img1 = Image.open('header.png')

# Display header and sidebar
st.header("Defect Classifier Model")
st.sidebar.image(img1)
st.sidebar.header("Predict Defect")


# Function to capture image from camera
def get_live_image():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    if ret:
        cap.release()
        return frame
    
# Function to transform video frame to RGB format
def transform(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# Function to predict image class
def predict(live_image):
    # Load model and label binarizer
    lb = pickle.load(open("lb.pkl","rb"))
    model = load_model("model.h5")

    # Resize image and convert to numpy array
    image = cv2.resize(live_image, (64, 64))
    image = np.array(image)
    image = np.reshape(image, (1, 64, 64, 3))

    # Make predictions and return result
    y_pred = model.predict(image)
    y_predicted_labels = [np.argmax(i) for i in y_pred]
    labels = lb.classes_
    label = labels[y_predicted_labels[0]]
    return label

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
    # Create video transformer to capture live video from camera
    webrtc_ctx = webrtc.StreamlitWebRTC(
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": False},
        key="camera",
    )
    webrtc_transformer = webrtc.VideoTransformer(
        source_video_transformer=None,
        transformer_factory=lambda: webrtc_opencv.VideoCaptureTransformer(transform=transform),
        source_transformer_factory=lambda: webrtc_ctx,
        fps=30,
        frame_size=(640, 480),
        allow_stream_capture=False,
    )

    # Prompt user to capture image
    if st.sidebar.button("Capture and Predict"):
        # Capture image from camera
        live_image = webrtc_transformer.last_frame

        # Check if image is captured
        if live_image is not None:
            st.image(live_image, caption="Live Image", use_column_width=True)

            # Make predictions and display result
            label = predict(live_image)
            st.subheader("Predicted Class : {}".format(label))
