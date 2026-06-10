import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# -------------------------------
# CONFIG PAGE
# -------------------------------
st.set_page_config(
    page_title="Waste Classification",
    layout="wide",
    page_icon="♻️"
)

# -------------------------------
# CUSTOM CSS (Fix Font Sizes)
# -------------------------------
st.markdown("""
<style>
    /* 1. Bigger font size for page main titles */
    .main-title {
        font-size: 56px !important;
        font-weight: 700 !important;
        margin-bottom: 20px !important;
    }
    /* 2. Smaller font size for subheaders (Problem, Objective, Dataset) */
    .section-header {
        font-size: 22px !important;
        font-weight: 600 !important;
        margin-top: 15px !important;
        margin-bottom: 10px !important;
    }
    /* 3. Larger navigation tab labels */
    button[role="tab"], div[role="tab"] {
        font-size: 20px !important;
        font-weight: 700 !important;
    }

    /* Primary button style for Predict Waste Category */
    .stButton>button, button[kind="primary"] {
        background-color: rgba(56, 130, 255, 0.16) !important;
        color: #0c3d91 !important;
        border: 1px solid rgba(56, 130, 255, 0.32) !important;
        box-shadow: none !important;
    }

    .bin-card {
        padding: 18px 22px;
        border-radius: 18px;
        color: #1f1f1f;
        font-weight: 700;
        font-size: 18px;
        margin-bottom: 16px;
        border: 1px solid rgba(0, 0, 0, 0.08);
        backdrop-filter: blur(8px);
    }
    .bin-card.hazardous {
        background: rgba(217, 83, 79, 0.16);
        color: #7a1f1f;
    }
    .bin-card.biohazard {
        background: rgba(142, 10, 16, 0.16);
        color: #6a0e12;
    }
    .bin-card.recyclable {
        background: rgba(26, 143, 47, 0.16);
        color: #155e20;
    }
    .bin-card.organic {
        background: rgba(240, 173, 78, 0.16);
        color: #5f430f;
    }
    .bin-card.general {
        background: rgba(108, 117, 125, 0.16);
        color: #2e3438;
    }
            
    /* Model Selectbox Header */
    .config-container {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #e9ecef;
        margin-bottom: 20px;
    }

    /* Bin Mapping UI Enhancement */
    .bin-card {
        padding: 18px;
        border-radius: 8px;
        font-size: 18px;
        font-weight: 600;
        text-align: center;
        margin-bottom: 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .hazardous { background-color: #fff0f0; color: #d63031; border: 1px solid #ffccd2; }
    .biohazard { background-color: #ffeaa7; color: #d63031; border: 1px solid #fdcb6e; }
    .recyclable { background-color: #e3f2fd; color: #0d47a1; border: 1px solid #bbdefb; }
    .organic { background-color: #e8f5e9; color: #1b5e20; border: 1px solid #c8e6c9; }
    .general { background-color: #f5f5f5; color: #424242; border: 1px solid #e0e0e0; }

    /* Handling Guide Cards */
    .action-card {
        padding: 15px;
        border-radius: 8px;
        margin-top: 15px;
        font-size: 15px;
        line-height: 1.5;
    }
    .action-recycle { background-color: #e8f8f5; border-left: 5px solid #2ecc71; color: #117a65; }
    .action-reuse { background-color: #fef9e7; border-left: 5px solid #f1c40f; color: #9a7d0a; }
    .action-compost { background-color: #f5eeeb; border-left: 5px solid #a04000; color: #5e2f0d; }
    .action-special { background-color: #fbeee6; border-left: 5px solid #e67e22; color: #78281f; }

</style>
""", unsafe_allow_html=True)

# -------------------------------
# CACHE MODEL
# -------------------------------
@st.cache_resource
def load_resnet():
    return tf.keras.models.load_model("resnet50_waste_classifier_v2.keras")

@st.cache_resource
def load_efficientnet():
    return tf.keras.models.load_model("efficientnet_waste_classifier_v4.keras")

@st.cache_resource
def load_mobilenet():
    return tf.keras.models.load_model("mobilenet_waste_classifier_v2.keras")

# -------------------------------
# DATA CONFIG & MAPPING
# -------------------------------
class_names = [
    'battery', 'biological', 'cardboard', 'clothes', 'glass',
    'metal', 'paper', 'plastic', 'shoes', 'trash'
]

bin_mapping = {
    'battery': 'Hazardous Waste',
    'trash': 'BioHazardous Waste',
    'cardboard': 'Recyclable Waste',
    'paper': 'Recyclable Waste',
    'glass': 'Recyclable Waste',
    'metal': 'Recyclable Waste',
    'plastic': 'Recyclable Waste',
    'biological': 'Organic Waste',
    'clothes': 'General Waste',
    'shoes': 'General Waste'
}

def preprocess_image(img):
    img = img.resize((224, 224))
    img_array = np.array(img, dtype=np.uint8)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array.astype(np.float32)

# -------------------------------
# Navigation 1: HOME
# -------------------------------
def home_page():
    # ---------------------------------------------------------
    # MAIN TITLE & BANNER
    # ---------------------------------------------------------
    st.markdown('<p class="main-title" style="text-align: center;">♻️ Waste Classification With Computer Vision</p>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        st.image(
            "assets/waste_header.jpg",
            use_container_width=True,
        )
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # ---------------------------------------------------------
    # SECTION 1: PROBLEM STATEMENT
    # ---------------------------------------------------------
    st.markdown(
        """
        <div style="background-color: #f9f9f9; padding: 20px; border-left: 5px solid #ff4b4b; border-radius: 5px; margin-bottom: 25px;">
            <p class="section-header" style="margin-top:0px; color: #ff4b4b;">📌 Problem Statement</p>
            <p style="font-size: 16px; line-height: 1.6; color: #333333; margin-bottom: 0px;">
                Many people throw waste into the wrong bins because sorting plastic, food, and batteries is confusing. 
                This mistake ruins recycling and fills up landfills quickly. Also, sorting trash by hand is very slow and 
                costs too much money. This project builds an AI app that uses a camera to sort waste quickly and accurately.
            </p>
        </div>
        """, 
        unsafe_allow_html=True
    )
    
    # ---------------------------------------------------------
    # SECTION 2: OBJECTIVES
    # ---------------------------------------------------------
    st.markdown(
        """
        <div style="background-color: #f9f9f9; padding: 20px; border-left: 5px solid #00bcff; border-radius: 5px; margin-bottom: 25px;">
            <p class="section-header" style="margin-top:0px; color: #00bcff;">🎯 Objectives</p>
            <p style="font-size: 16px; line-height: 1.6; color: #333333; margin-bottom: 10px;">
                This repository contains computer vision experiments, trained models, and a Streamlit inference app for classifying waste into 10 categories.
            </p>
            <p style="font-size: 16px; line-height: 1.6; color: #333333; margin-bottom: 0px; font-weight: 500;">
                💡 Features transfer learning with <b>ResNet50</b>, <b>EfficientNet-B0</b>, and <b>MobileNetV2</b> to build practical, deployable waste classifiers.
            </p>
        </div>
        """, 
        unsafe_allow_html=True
    )
    
    # ---------------------------------------------------------
    # SECTION 3: DATASET OVERVIEW 
    # ---------------------------------------------------------
    st.markdown('<p class="section-header" style="color: #2ed573;">📊 Dataset Overview</p>', unsafe_allow_html=True)
  
    st.write(
        "The dataset includes images of household waste separated into classes such as paper, cardboard, "
        "metal, plastic, glass, clothes, shoes, battery, biological waste, and trash. In this project, "
        "glass subcategories are merged into a single glass category."
    )
    
    st.info(
        "📈 **Dataset Scale:** The complete dataset consists of **15,500 total images** used during the training and validation phases. "
        "All images are resized to **224x224 pixels** to optimize feature extraction across all deep learning networks.\n\n"
        "🔗 **Source Link:** You can explore and download the original dataset directly from the "
        "[Kaggle Waste Classification Dataset](https://www.kaggle.com/datasets/sumn2u/garbage-classification-v2/data)."
    )
    
    classes_badges = " ".join([f'<span style="background-color:#e1f5fe; color:#0288d1; padding:4px 10px; margin:4px; border-radius:15px; display:inline-block; font-size:14px; font-weight:500;">{c.title()}</span>' for c in class_names])
    st.markdown(f"<div style='margin-bottom: 25px;'><b>Target Classes:</b> {classes_badges}</div>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # ---------------------------------------------------------
    # SECTION 4: DYNAMIC IMAGE GRID EXAMPLES
    # ---------------------------------------------------------
    st.markdown('<p class="section-header" style="color: #2f3542; font-size: 20px !important;">🖼️ Dataset Class Examples</p>', unsafe_allow_html=True)
    st.write("Click on each folder dropdown below to view actual samples included in the training categories:")
    
    # Grid layout: 5 main columns
    grid_cols = st.columns(5)
    
    for index, class_item in enumerate(class_names):
        col_to_use = grid_cols[index % 5]
        
        with col_to_use:
            # Styled expander container
            with st.expander(f"📁 {class_item.title()}", expanded=True):
                class_folder_path = f"assets/example/{class_item}"
                
                if os.path.exists(class_folder_path) and os.path.isdir(class_folder_path):
                    all_images = [
                        f for f in os.listdir(class_folder_path) 
                        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
                    ]
                    
                    if all_images:
                     
                        num_imgs = len(all_images)
                        sub_cols = st.columns(min(num_imgs, 2)) # Displays max 2 items per row inside expander
                        
                        for img_idx, img_file in enumerate(all_images):
                            sub_col_idx = img_idx % min(num_imgs, 2)
                            full_img_path = os.path.join(class_folder_path, img_file)
                            
                            with sub_cols[sub_col_idx]:
                                # Clean borderless square crops for modern thumbnail look
                                st.image(
                                    full_img_path, 
                                    use_container_width=True
                                )
                    else:
                        st.markdown("<p style='color:#888888; font-size:12px; font-style:italic; text-align:center;'>No samples found</p>", unsafe_allow_html=True)
                else:
                    # Clean Fallback UI using standard placeholder API
                    placeholder_url = f"https://placehold.co{class_item.title()}"
                    st.image(placeholder_url, use_container_width=True)



# -------------------------------
# Navigation 2: PREDICTION
# -------------------------------
def prediction_page():
    st.markdown('<p class="main-title" style="text-align: center;">Waste Category Prediction</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1], gap="large")
    image_source = None

    with col1:
        st.subheader("Model Selection")
        selected_model_name = st.selectbox(
            "Choose Deep Learning Architecture",
            ["ResNet-50", "EfficientNet-B0", "MobileNetV2"],
            label_visibility="collapsed"
        )
        
        if selected_model_name == "ResNet-50":
            model = load_resnet()
        elif selected_model_name == "EfficientNet-B0":
            model = load_efficientnet()
        else:
            model = load_mobilenet()
            
        st.success(f"🚀 **Active Model:** {selected_model_name} successfully loaded.")
        st.markdown("---")

        st.subheader("📸 Input Image")
        tab1, tab2 = st.tabs(["📤 Upload File", "📷 Use Webcam"])

        with tab1:
            uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"], key="uploader")
            if uploaded_file:
                image_source = Image.open(uploaded_file).convert("RGB")

        with tab2:
            camera_file = st.camera_input("Capture Image", key="camera")
            if camera_file:
                image_source = Image.open(camera_file).convert("RGB")

        if image_source:
            st.image(image_source, caption="Selected Waste Sample", use_container_width=False)
            predict_clicked = st.button("Predict Waste Category", use_container_width=False, type="primary")
            
            if predict_clicked:
                with st.spinner("Analyzing image patterns and extracting features..."):
                    img_array = preprocess_image(image_source)
                    predictions = model.predict(img_array, verbose=0)

                    predicted_index = int(np.argmax(predictions[0]))
                    predicted_class = class_names[predicted_index]
                    confidence = float(np.max(predictions[0])) * 100
                    assigned_bin = bin_mapping.get(predicted_class, "General Waste")

                    st.session_state.prediction = {
                        "predicted_class": predicted_class,
                        "confidence": confidence,
                        "assigned_bin": assigned_bin,
                        "predictions": predictions[0].tolist(),
                    }
        else:
            st.info("💡 Pro-Tip: Please upload or capture an image to execute inference.")
            st.session_state.prediction = None

    with col2:
        st.subheader("Prediction Results")

        if st.session_state.prediction is None:
            st.warning("⚠️ Awaiting input image selection and prediction trigger.")
        else:
            res = st.session_state.prediction
            pred_class_lower = res["predicted_class"].lower()

            # Clean Minimalist Metric Cards
            m_col1, m_col2 = st.columns(2)
            with m_col1:
                st.metric("Predicted as:", res["predicted_class"].upper())
            with m_col2:
                st.metric("Model Confidence", f"{res['confidence']:.2f}%")

            st.subheader('♻️ Recommended Disposal Bin')

            bin_color_map = {
                'Hazardous Waste': 'hazardous',
                'Biological Hazardous Waste': 'biohazard',
                'Recyclable Waste': 'recyclable',
                'Organic Waste': 'organic',
                'General Waste': 'general',
            }
            bin_css_class = bin_color_map.get(res['assigned_bin'], 'general')
            
            # FIXED ICON: Replaced all individual icons with standard recycle icon ♻️
            st.markdown(
                f'<div class="bin-card {bin_css_class}">{res["assigned_bin"].upper()}</div>',
                unsafe_allow_html=True,
            )

            # ---------------------------------------------------------
            #  ACTION GUIDE
            # ---------------------------------------------------------
            st.subheader("💡 Suggested Waste Handling Action")
            
            if pred_class_lower in ['cardboard', 'paper', 'glass', 'metal', 'plastic']:
                st.markdown(
                    """
                    <div class="action-card action-recycle">
                        <strong>♻️ Action: High Recycling Value</strong><br>
                        This material should be sorted and placed into the recycling stream. Please empty and rinse any liquid or food residues before throwing it away to avoid contamination.
                    </div>
                    """, unsafe_allow_html=True
                )
            elif pred_class_lower in ['clothes', 'shoes']:
                st.markdown(
                    """
                    <div class="action-card action-reuse">
                        <strong>👕 Action: Reuse & Donation Candidate</strong><br>
                        Do not discard in standard trash bins if still usable. Consider donating to local charity organizations, clothing banks, or sending them to specialized fabric recycling centers.
                    </div>
                    """, unsafe_allow_html=True
                )
            elif pred_class_lower == 'biological':
                st.markdown(
                    """
                    <div class="action-card action-compost">
                        <strong>🍂 Action: Organic Composting</strong><br>
                        This organic matter decomposes naturally. Highly recommended to process inside a domestic backyard compost bin or send to municipal bio-waste systems to produce nutrient-rich fertilizer.
                    </div>
                    """, unsafe_allow_html=True
                )
            elif pred_class_lower in ['battery', 'trash']:
                st.markdown(
                    """
                    <div class="action-card action-special">
                        <strong>⚠️ Action: Special Chemical/Biological Handling</strong><br>
                        This item contains dangerous materials or bio-hazards. Never mix with regular household waste. Must be brought to a specialized municipal chemical collection site or e-waste recycling collection drop-off.
                    </div>
                    """, unsafe_allow_html=True
                )

            st.markdown("---")
            
            # Top 3 Probabilities Section
            st.subheader("📈 Top 3 Probabilities")
            top3_idx = np.array(res["predictions"]).argsort()[-3:][::-1]
            for idx in top3_idx:
                prob = res["predictions"][idx]
                class_label = class_names[idx].title()
                
                st.write(f"**{class_label}** — *{prob * 100:.1f}% Match*")
                st.progress(float(prob))

# ==============================================================================
# 🎯 CORE NAVIGATION IN UI (Urutan: Title -> Navigation)
# ==============================================================================
pg = st.navigation(
    pages=[
        st.Page(home_page, title="Home", icon="🏠"),
        st.Page(prediction_page, title="Prediction", icon="🔎")
        
    ],
    position="sidebar",
)

# Menjalankan logik bagi memuatkan kandungan halaman yang sedang aktif
pg.run()



# ==============================================================================
# 📝 SIDEBAR BOTTOM: HOW TO USE CARD (Urutan: -> How to Use)
# ==============================================================================

st.sidebar.markdown(
    """
    <div style="
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        padding: 18px; 
        border-radius: 12px; 
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.03);
        font-family: 'Inter', sans-serif;
    ">
        <p style="font-size: 16px; font-weight: 700; color: #0f172a; margin-top: 0px; margin-bottom: 14px;">
            📝 How to Use
        </p>
        <div style="font-size: 13px; color: #334155; line-height: 1.6;">
            <div style="margin-bottom: 11px; display: flex; gap: 8px;">
                <span style="background: #3882ff; color: white; border-radius: 50%; width: 20px; height: 20px; display: inline-flex; align-items: center; justify-content: center; font-size: 11px; font-weight: bold; flex-shrink: 0; margin-top: 2px;">1</span>
                <span>Select the <b>Prediction</b> page from the menu above.</span>
            </div>
            <div style="margin-bottom: 11px; display: flex; gap: 8px;">
                <span style="background: #3882ff; color: white; border-radius: 50%; width: 20px; height: 20px; display: inline-flex; align-items: center; justify-content: center; font-size: 11px; font-weight: bold; flex-shrink: 0; margin-top: 2px;">2</span>
                <span>Choose a <b>Deep Learning model</b> from the dropdown list.</span>
            </div>
            <div style="margin-bottom: 11px; display: flex; gap: 8px;">
                <span style="background: #3882ff; color: white; border-radius: 50%; width: 20px; height: 20px; display: inline-flex; align-items: center; justify-content: center; font-size: 11px; font-weight: bold; flex-shrink: 0; margin-top: 2px;">3</span>
                <span><b>Upload</b> an image or capture one live using your <b>Webcam</b>.</span>
            </div>
            <div style="margin-bottom: 11px; display: flex; gap: 8px;">
                <span style="background: #3882ff; color: white; border-radius: 50%; width: 20px; height: 20px; display: inline-flex; align-items: center; justify-content: center; font-size: 11px; font-weight: bold; flex-shrink: 0; margin-top: 2px;">4</span>
                <span>Click the <b>Predict Waste Category</b> button.</span>
            </div>
            <div style="display: flex; gap: 8px;">
                <span style="background: #3882ff; color: white; border-radius: 50%; width: 20px; height: 20px; display: inline-flex; align-items: center; justify-content: center; font-size: 11px; font-weight: bold; flex-shrink: 0; margin-top: 2px;">5</span>
                <span>View the <b>results, bin recommendation, and charts</b> on the right.</span>
            </div>
        </div>
    </div>
    """, 
    unsafe_allow_html=True
)




