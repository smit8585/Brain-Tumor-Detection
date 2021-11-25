import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# def
def show_uploaded_img(uploaded_file):
  img = Image.open(uploaded_file)
  st.subheader("Uploaded file")
  st.image(img)
  return img

def names(number):
    if number==0:
        return 'Its a Tumor'
    else:
        return 'Its NOT a tumor'

def make_prediction(img):
  x = np.array(img.resize((128,128)))
  x = x.reshape(1,128,128,3)
  res = modelx.predict_on_batch(x)
  classification = np.where(res == np.amax(res))[1][0]
  st.subheader("Results")
  st.text(str(res[0][classification]*100) + '% Confident that ' +names(classification))
  #st.text(res) 

st.title("Brain Tumour Classifier")

st.header("Identify Tumors in MRI / CT Scan Images")

st.subheader("Upload a file Brain scanned file")

uploaded_file = st.file_uploader("", type=['png','jpg','jpeg'])

modelx=tf.keras.models.load_model('my_model.hdf5') #getting pretrained model

if uploaded_file is not None:
  img = show_uploaded_img(uploaded_file)
  if st.button("Classify"):
    make_prediction(img)
  #else:
   # st.write("Some error occured while Classifying the image!!!")
else:
  st.write("No image uploaded!!!")
  st.write("Upload an image file to identify tumor.")
