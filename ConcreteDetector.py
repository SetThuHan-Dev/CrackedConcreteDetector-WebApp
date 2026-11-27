# coding=utf8
import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.utils import img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import load_model

st.title("Cracked Or Uncracked 🤔❓")
st.header("Please upload any concrete image 🖌️ (.jpg)")
uploaded_file = st.file_uploader("Upload a concrete image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")


@st.cache_resource(show_spinner="Loading VGG16 model… ⏳")
def get_model():
    return load_model('classifier_vgg16_model.h5', compile=False)


if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='*Uploaded Image*', use_container_width=True)
    st.write("")

    vgg16_model = get_model()

    # Resize to VGG16 expected input size (224x224)
    img_resized = image.resize((224, 224))
    img_array = img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    prediction = vgg16_model.predict(img_array)
    predicted_class = np.argmax(prediction[0])

    if predicted_class == 0:
        st.success("Uncracked Concrete 🧱")
    else:
        st.error("Cracked Concrete! 🚧")
        st.info("Info: Why does concrete crack? For more information, please Google it! 🤪")
