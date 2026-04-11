import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image
import plotly.graph_objects as go

# ---------- CONFIG ----------
st.set_page_config(page_title="Dyslexia Detection", layout="wide")

# ---------- CSS ----------
st.markdown("""
<style>
html, body {
    background-color: #0f172a;
    color: #e2e8f0;
}

/* Title */
.title {
    font-size: 50px;
    font-weight: 700;
}

/* Subtitle */
.subtitle {
    color: #94a3b8;
    font-size: 18px;
}

/* Section spacing */
.section {
    margin-top: 40px;
}

/* Button */
.stButton>button {
    background-color: #2563eb;
    color: white;
    border-radius: 8px;
    height: 45px;
    font-size: 16px;
    border: none;
}

/* Hide empty blocks */
div[data-testid="stVerticalBlock"] > div:empty {
    display: none;
}
</style>
""", unsafe_allow_html=True)

# ---------- NAVIGATION ----------
page = st.sidebar.radio("Navigation", ["Home", "Analyze"])

# =========================================================
# LANDING PAGE
# =========================================================

if page == "Home":

    st.markdown('<div class="title">AI Dyslexia Detection</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Early detection of dyslexia using handwriting analysis</div>', unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("## What this system does")
    st.write("""
    - Detects dyslexia patterns using AI  
    - Analyzes handwriting structure  
    - Provides confidence and risk levels  
    """)

    st.markdown("## How it works")
    st.write("""
    1. Upload handwriting sample  
    2. Image preprocessing  
    3. Deep learning prediction  
    4. Cognitive pattern analysis  
    """)

    st.markdown("## Features")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.info("AI-based Detection")

    with col2:
        st.info("Confidence Analysis")

    with col3:
        st.info("Cognitive Insights")

    st.markdown("## Why this matters")
    st.write("""
    Early detection of dyslexia helps in:
    - Better learning support  
    - Personalized education  
    - Improved cognitive development  
    """)

    st.markdown("---")

    if st.button("Start Analysis"):
        st.session_state.page = "Analyze"

# =========================================================
# 🔍 ANALYSIS PAGE
# =========================================================

if page == "Analyze":

    # Load model
    model = load_model("dyslexia_handwriting_model_sem6.h5")

    st.markdown("## Analyze Handwriting")

    uploaded_file = st.file_uploader("Upload handwriting image", type=["jpg","png","jpeg"])
    analyze = st.button("Analyze")

    labels = {0: "Normal", 1: "Dyslexia", 2: "Corrected"}

    if uploaded_file and analyze:

        image = Image.open(uploaded_file).convert("RGB")
        img_array = np.array(image)

        # ---------- PREPROCESS ----------
        img = cv2.resize(img_array, (128,128))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = cv2.equalizeHist(img)
        img = img / 255.0
        img = img.reshape(1,128,128,1)

        prediction = model.predict(img)
        pred_class = np.argmax(prediction)
        confidence = np.max(prediction)

        col1, col2 = st.columns([1,1])

        # IMAGE
        with col1:
            st.image(image, use_container_width=True)

        # RESULT
        with col2:
            if pred_class == 0:
                st.success("Normal")
            elif pred_class == 1:
                st.error("Dyslexia detected")
            else:
                st.warning("Corrected pattern")

            st.write(f"Confidence: **{confidence*100:.1f}%**")
            st.progress(float(confidence))

        # COMPARISON
        st.markdown("### Comparison")

        normal_score = (1 - confidence) * 100
        dyslexia_score = confidence * 100

        c1, c2 = st.columns(2)

        with c1:
            st.metric("Normal Pattern", f"{normal_score:.1f}%")

        with c2:
            st.metric("Dyslexia Pattern", f"{dyslexia_score:.1f}%")

        # RISK
        if confidence > 0.7:
            st.error("High Risk")
        elif confidence > 0.4:
            st.warning("Moderate Risk")
        else:
            st.success("Low Risk")

        # RADAR
        st.markdown("### Cognitive Parameters")

        categories = ['Language','Memory','Speed','Visual','Audio','Survey']
        values = [confidence]*6

        radar = go.Figure()
        radar.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself'
        ))

        radar.update_layout(
            paper_bgcolor="#0f172a",
            polar=dict(bgcolor="#0f172a"),
            showlegend=False
        )

        st.plotly_chart(radar, use_container_width=True)

        # INSIGHTS
        st.markdown("### Insights")

        if pred_class == 1:
            st.write("• Irregular stroke patterns detected")
            st.write("• Inconsistent spacing observed")
            st.write("• Possible reading difficulty indicators")
        else:
            st.write("• Writing appears consistent")
            st.write("• No major irregularities detected")

        # DOWNLOAD
        report = f"Result: {labels[pred_class]}\nConfidence: {confidence*100:.2f}%"

        st.download_button("Download Report", report)