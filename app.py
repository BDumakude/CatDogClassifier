import streamlit as st
import numpy as np
from tensorflow.keras import models
from PIL import Image
import warnings

warnings.filterwarnings("ignore")

# ---------- Model loading (cached & safe) ----------
@st.cache_resource
def load_model():
    return models.load_model("cats_and_dogs_1.keras")

model = load_model()

# ---------- App layout ----------
header = st.container()
uploader = st.container()
viewer = st.container()
response = st.container()

# ---------- Helper functions ----------
def img_to_array(img):
    image = Image.open(img).convert("RGB").resize((150, 150))
    image = np.array(image) / 255.0
    image = image.reshape((1,) + image.shape)
    return image

def prediction(img_array):
    pred = model.predict(img_array, verbose=0)[0][0]
    return "Dog" if pred >= 0.5 else "Cat"

# ---------- UI ----------
with header:
    st.title("Cat or Dog")

with uploader:
    img = st.file_uploader("Upload Picture", type=["jpg", "png", "jpeg"])

if img is not None:
    with viewer:
        col1, col2, col3 = st.columns(3)
        with col2:
            st.image(img, caption="Your Image", use_container_width=True)

with response:
    if st.button("Classify"):
        if img is None:
            st.warning("Please upload an image first.")
        else:
            img_array = img_to_array(img)
            result = prediction(img_array)
            st.success(f"This is a **{result}**!")
