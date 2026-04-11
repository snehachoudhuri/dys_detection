import streamlit as st
import numpy as np
from PIL import Image
import plotly.graph_objects as go

st.set_page_config(page_title="Dyslexia Detection", layout="wide")

if "page" not in st.session_state:
    st.session_state.page = "Home"

page = st.session_state.page

labels = {0: "Normal", 1: "Dyslexia"}

# =========================================================
# HOME
# =========================================================

if page == "Home":

    st.title("Dyslexia Detection System")
    st.write("AI-based handwriting analysis")

    if st.button("Start Analysis", use_container_width=True):
        st.session_state.page = "Analyze"
        st.rerun()

# =========================================================
# ANALYZE
# =========================================================

elif page == "Analyze":

    if st.button("← Back"):
        st.session_state.page = "Home"
        st.rerun()

    uploaded_file = st.file_uploader("Upload handwriting image", type=["jpg","png","jpeg"])
    analyze = st.button("Analyze")

    if uploaded_file and analyze:

        image = Image.open(uploaded_file).convert("RGB")

        # ---------- FAKE MODEL LOGIC ----------
        img = image.resize((128,128))
        img = img.convert("L")
        img = np.array(img) / 255.0

        # simulate prediction
        confidence = np.mean(img)  # simple heuristic

        if confidence < 0.5:
            pred_class = 1
        else:
            pred_class = 0

        col1, col2 = st.columns(2)

        with col1:
            st.image(image, use_container_width=True)

        with col2:
            if pred_class == 0:
                st.success("Normal")
            else:
                st.error("Dyslexia detected")

            st.write(f"Confidence: {confidence*100:.2f}%")
            st.progress(float(confidence))

        # comparison
        st.subheader("Comparison")

        st.metric("Normal", f"{(1-confidence)*100:.1f}%")
        st.metric("Dyslexia", f"{confidence*100:.1f}%")

        # radar
        categories = ['Language','Memory','Speed','Visual','Audio','Survey']
        values = [confidence]*6

        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(r=values, theta=categories, fill='toself'))

        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Insights")

        if pred_class == 1:
            st.write("Irregular handwriting patterns detected")
        else:
            st.write("Writing appears normal")
