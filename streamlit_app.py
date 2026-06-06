import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(
    page_title="Smart Waste Classifier",
    layout="wide",
    page_icon="♻️"
)

# -------------------------------
# LOAD MODELS
# -------------------------------
@st.cache_resource
def load_resnet():
    return tf.keras.models.load_model("resnet50_waste_classifier_v2.keras")

@st.cache_resource
def load_efficientnet():
    return tf.keras.models.load_model("efficientnet_waste_classifier_v3.keras")

@st.cache_resource
def load_mobilenet():
    return tf.keras.models.load_model("mobilenet_waste_classifier_v2.keras")

# -------------------------------
# CLASS LABELS
# -------------------------------
class_names = [
    'battery', 'biological', 'cardboard', 'clothes', 'glass',
    'metal', 'paper', 'plastic', 'shoes', 'trash'
]

# -------------------------------
# BIN MAPPING
# -------------------------------
bin_mapping = {
    'battery': 'Hazardous Waste',
    'trash': 'Biological Hazardous Waste',
    'cardboard': 'Recyclable Waste',
    'paper': 'Recyclable Waste',
    'glass': 'Recyclable Waste',
    'metal': 'Recyclable Waste',
    'plastic': 'Recyclable Waste',
    'biological': 'Organic Waste',
    'clothes': 'General Waste',
    'shoes': 'General Waste'
}

# -------------------------------
# UI
# -------------------------------
st.title("♻️ Waste Classifier")

st.sidebar.header("⚙️ Model Selection")

selected_model_name = st.sidebar.radio(
    "Choose Model",
    ["ResNet-50", "EfficientNet-B0", "MobileNetV2"]
)

# -------------------------------
# LOAD MODEL
# -------------------------------
if selected_model_name == "ResNet-50":
    model = load_resnet()

elif selected_model_name == "EfficientNet-B0":
    model = load_efficientnet()

else:
    model = load_mobilenet()

st.sidebar.success(f"Active Model: {selected_model_name}")

# -------------------------------
# IMAGE INPUT
# -------------------------------
col1, col2 = st.columns(2)

image_source = None

with col1:
    st.subheader("📸 Input Image")

    tab1, tab2 = st.tabs(["Upload", "Webcam"])

    with tab1:
        uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            image_source = Image.open(uploaded_file).convert("RGB")

    with tab2:
        camera_file = st.camera_input("Capture Image")
        if camera_file:
            image_source = Image.open(camera_file).convert("RGB")

    if image_source:
        st.image(image_source, caption="Input Image", use_container_width=True)

# -------------------------------
# PREPROCESSING 
# -------------------------------

def preprocess_image(img, model_name):

    img = img.resize((224, 224))
    
    img_array = np.array(img,dtype=np.uint8)
    img_array = np.expand_dims(img_array, axis=0)

  
    # if model_name == "MobileNetV2":

    #     return img_array
        
    # elif model_name == "ResNet-50":

    #     return img_array.astype(np.float32)
        
    # elif model_name == "EfficientNet-B0":

    #     return img_array.astype(np.float32)

    return img_array.astype(np.float32)

# -------------------------------
# PREDICTION
# -------------------------------
with col2:
    st.subheader("📊 Prediction Result")

    if image_source is not None:

        if st.button("✨ Predict", use_container_width=True):

            with st.spinner("Predicting..."):

                img_array = preprocess_image(
                    image_source,
                    selected_model_name,
                    
                )

                predictions = model.predict(img_array, verbose=0)

                predicted_index = int(np.argmax(predictions[0]))
                predicted_class = class_names[predicted_index]
                confidence = float(np.max(predictions)) * 100

                assigned_bin = bin_mapping.get(predicted_class, "General Waste")

                # -------------------------------
                # OUTPUT
                # -------------------------------
                st.metric("Predicted Class", predicted_class.upper())
                st.metric("Confidence", f"{confidence:.2f}%")

                st.write("---")
                st.subheader("🗑️ Bin Recommendation")
                st.success(f"Assigned Bin: {assigned_bin}")

                # -------------------------------
                # TOP 3
                # -------------------------------
                st.subheader("📊 Top 3 Predictions")

                top3 = predictions[0].argsort()[-3:][::-1]

                for i in top3:
                    st.write(f"{class_names[i]}: {predictions[0][i]:.3f}")