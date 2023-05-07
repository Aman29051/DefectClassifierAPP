import streamlit as st
from PIL import Image
import cv2
import pickle
import io
import numpy as np
from keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

st.set_page_config(page_title="Defect Classifier",page_icon="memo",layout="wide")

img1 = Image.open('header.png')

st.header("Defect Classifier Model")

st.sidebar.image(img1)
st.sidebar.header("Predict Defect")
inp_image = st.sidebar.file_uploader("Choose Input Image",type=["jpg","png","jpeg","bmp"])

if inp_image:
    # Read file contents into memory
    image_bytes = inp_image.read()

    # Create PIL Image object
    image_ = Image.open(io.BytesIO(image_bytes))
    st.image(image_,caption="Input Image")
    
    # Load model and label binarizer
    lb = pickle.load(open("lb.pkl","rb"))
    model = load_model("model.h5")
    
    # Create numpy array from file buffer
    image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), 1)
    image = cv2.resize(image,(64,64))
    image = np.array(image)
    print(image.shape)
    image = np.reshape(image,(1,64,64,3))

    # Make predictions and display result
    y_pred = model.predict(image)
    y_predicted_labels = [np.argmax(i) for i in y_pred]
    labels = lb.classes_
    label = labels[y_predicted_labels[0]]
    if st.sidebar.button("Predict Defect"):
        st.subheader("Predicted Class : {}".format(label))
