import streamlit as st
import numpy as np
from keras.models import load_model
from PIL import Image
import plotly.graph_objects as go

# ---------- CONFIG ----------
st.set_page_config(page_title="Dyslexia Detection", layout="wide")

# ---------- PAGE STATE ----------
if "page" not in st.session_state:
    st.session_state.page = "Home"

page = st.session_state.page

labels = {0: "Normal", 1: "Dyslexia", 2: "Corrected"}

# ---------- LOAD MODEL ----------
model = load_model("dyslexia_handwriting_model_sem6.h5")

# =========================================================
# 🏠 HOME PAGE
# =========================================================

if page == "Home":

    st.title("Dyslexia Detection System")

    st.write("AI-based handwriting analysis to detect dyslexia patterns.")

    st.markdown("---")

    st.write("### Features")
    st.write("- Detect handwriting patterns")
    st.write("- Confidence analysis")
    st.write("- Cognitive insights")

    st.markdown("---")

    if st.button("Start Analysis", use_container_width=True):
        st.session_state.page = "Analyze"
        st.rerun()

# =========================================================
# 🔍 ANALYZE PAGE
# =========================================================

elif page == "Analyze":

    if st.button("← Back"):
        st.session_state.page = "Home"
        st.rerun()

    st.header("Analyze Handwriting")

    uploaded_file = st.file_uploader("Upload handwriting image", type=["jpg","png","jpeg"])
    analyze = st.button("Analyze")

    if uploaded_file and analyze:

        image = Image.open(uploaded_file).convert("RGB")

        # ---------- PREPROCESS (NO OPENCV) ----------
        image = image.resize((128,128))
        image = image.convert("L")  # grayscale
        img = np.array(image)
        img = img / 255.0
        img = img.reshape(1,128,128,1)

        # ---------- PREDICT ----------
        prediction = model.predict(img)
        pred_class = np.argmax(prediction)
        confidence = np.max(prediction)

        col1, col2 = st.columns([1,1])

        # SHOW IMAGE
        with col1:
            st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

        # RESULT
        with col2:
            if pred_class == 0:
                st.success("Normal")
            elif pred_class == 1:
                st.error("Dyslexia detected")
            else:
                st.warning("Corrected")

            st.write(f"Confidence: {confidence*100:.2f}%")
            st.progress(float(confidence))

        st.markdown("---")

        # ---------- COMPARISON ----------
        st.subheader("Comparison")

        normal_score = (1 - confidence) * 100
        dyslexia_score = confidence * 100

        c1, c2 = st.columns(2)

        with c1:
            st.metric("Normal Pattern", f"{normal_score:.1f}%")

        with c2:
            st.metric("Dyslexia Pattern", f"{dyslexia_score:.1f}%")

        # ---------- RISK ----------
        if confidence > 0.7:
            st.error("High Risk")
        elif confidence > 0.4:
            st.warning("Moderate Risk")
        else:
            st.success("Low Risk")

        st.markdown("---")

        # ---------- RADAR ----------
        st.subheader("Cognitive Parameters")

        categories = ['Language','Memory','Speed','Visual','Audio','Survey']
        values = [confidence]*6

        radar = go.Figure()
        radar.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself'
        ))

        radar.update_layout(
            showlegend=False
        )

        st.plotly_chart(radar, use_container_width=True)

        # ---------- INSIGHTS ----------
        st.subheader("Insights")

        if pred_class == 1:
            st.write("• Irregular writing patterns detected")
            st.write("• Spacing inconsistency observed")
            st.write("• Possible dyslexia indicators present")
        else:
            st.write("• Writing appears consistent")
            st.write("• No strong dyslexia indicators")

        # ---------- DOWNLOAD ----------
        report = f"""
        Result: {labels[pred_class]}
        Confidence: {confidence*100:.2f}%
        """

        st.download_button("Download Report", report)
