import streamlit as st 
import numpy as np 
from keras import models
from PIL import Image 
import warnings

warnings.filterwarnings('ignore')

model = models.load_model('cats_and_dogs_1.keras')

header = st.container()
uploader = st.container()
viewer = st.container()
response = st.container()

def img_to_array(img):
    image = np.array(Image.open(img).resize((150,150)))
    image_array = image.reshape((1,) + image.shape)
    return image_array 

def prediction(img_array):
    prediction = round(model.predict(img_array)[0,0])
    if prediction == 0:
        return "Cat"
    else:
        return "Dog"


with header:
    st.title("Cat or Dog")

with uploader:
    img = st.file_uploader("Upload Picture", type=['jpg','png'])

if img is not None:
    with viewer:    
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write("")

        with col2:
            st.image(image=img, caption="Your Image")

        with col3:
            st.write("")
    
if st.button("Classify"):
    img_as_array = img_to_array(img)
    final_decision = prediction(img_as_array)
    st.text(f"              This is a {final_decision}!")