import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.applications.efficientnet import preprocess_input

st.set_page_config(
    page_title="Pneumonia Detection",
    page_icon="Pneumonia",
    layout="centered"
)

st.markdown("""
<style>

[data-testid="stAppViewContainer"] {
    background-color: #0f172a;
}

[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] h1,
[data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3,
[data-testid="stText"] {
    color: white;
}

[data-testid="stFileUploader"] {
    background-color: #1e293b;
    border-radius: 10px;
    padding: 10px;
}

.stButton>button {
    background-color: #22c55e;
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
}

</style>
""", unsafe_allow_html=True)

st.markdown(
    "<h1 style='text-align: center; color: #22c55e;'>Pneumonia Detection App</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align: center;'>Upload a chest X-ray image to detect Pneumonia</p>",
    unsafe_allow_html=True
)

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model.h5")

model = load_model()

uploaded_file = st.file_uploader(
    "Upload X-ray Image",
    type=["jpg", "png", "jpeg"]
)

if uploaded_file is not None:

    img = Image.open(uploaded_file).convert("RGB")

    st.image(img, caption="Uploaded X-ray", use_container_width=True)

    img_resized = img.resize((224, 224))
    img_array = np.array(img_resized)

    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    with st.spinner("Analyzing X-ray..."):
        prediction = model.predict(img_array)[0][0]

    if prediction > 0.5:
        label = "Pneumonia"
        confidence = prediction
    else:
        label = "Normal"
        confidence = 1 - prediction

    st.markdown("---")

    if label == "Pneumonia":
        st.error(f"Pneumonia Detected ({confidence * 100:.2f}%)")
    else:
        st.success(f"Normal ({confidence * 100:.2f}%)")

    st.markdown("---")

    st.subheader("Prediction Breakdown")

    st.write("Pneumonia Probability")
    st.progress(float(prediction))

    st.write("Normal Probability")
    st.progress(float(1 - prediction))

    if confidence < 0.7:
        st.warning("Low confidence prediction. Model is unsure.")
    else:
        st.success("High confidence prediction.")
