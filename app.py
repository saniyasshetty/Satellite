"""
app.py
------
Streamlit web app: upload a satellite image, see the predicted poverty
level, confidence scores, and which visual features drove the prediction.

Run:
    streamlit run app.py
"""

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from features import extract_features, FEATURE_NAMES
from data_generator import generate_image, CLASSES as GEN_CLASSES

MODEL_PATH = "models/poverty_model.joblib"

st.set_page_config(page_title="Satellite Poverty Predictor", page_icon="🛰️", layout="centered")


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)


def predict_image(img: Image.Image, bundle):
    clf = bundle["model"]
    feats = extract_features(img).reshape(1, -1)
    pred = clf.predict(feats)[0]
    proba = clf.predict_proba(feats)[0]
    classes = clf.classes_
    return pred, dict(zip(classes, proba)), feats[0]


st.title("🛰️ Satellite Image-Based Poverty Prediction")
st.caption(
    "Upload a satellite-style image and the model estimates whether the area "
    "shown has LOW, MEDIUM, or HIGH poverty based on visible buildings, roads, "
    "and vegetation."
)

with st.expander("ℹ️ About this demo (read me)"):
    st.markdown(
        """
This project's model is trained on **procedurally generated synthetic satellite
images** (see `data_generator.py`), not real satellite imagery, because real
datasets (Sentinel-2 imagery + DHS poverty survey labels) require external
registration/API access not available in this build environment.

The generator controls building density, road count, and vegetation cover
per class, mimicking the real visual signals researchers use for this task
(see Jean et al. 2016, *"Combining satellite imagery and machine learning
to predict poverty"*, Science).

**To use real data:** replace `data_generator.py`'s output with real
satellite tiles + real DHS wealth-index labels, sorted into
`data/train/<low|medium|high>/` — nothing else in the pipeline needs to change.
        """
    )

bundle = load_model()
if bundle is None:
    st.error("No trained model found. Run `python train.py` first, then restart this app.")
    st.stop()

tab1, tab2 = st.tabs(["📤 Upload your own image", "🎲 Try a generated example"])

result_img = None
source_label = None

with tab1:
    uploaded = st.file_uploader("Upload a satellite/aerial image (PNG/JPG)", type=["png", "jpg", "jpeg"])
    if uploaded is not None:
        result_img = Image.open(uploaded).convert("RGB")
        source_label = uploaded.name

with tab2:
    st.write("No satellite image handy? Generate a random synthetic example area:")
    gen_class = st.selectbox("Pick a ground-truth class to generate", GEN_CLASSES, index=1)
    if st.button("🎲 Generate random area"):
        result_img = generate_image(gen_class)
        source_label = f"synthetic ({gen_class})"
        st.session_state["gen_img"] = result_img
        st.session_state["gen_label"] = source_label
    elif "gen_img" in st.session_state:
        result_img = st.session_state["gen_img"]
        source_label = st.session_state["gen_label"]

if result_img is not None:
    col1, col2 = st.columns([1, 1.3])
    with col1:
        st.image(result_img.resize((256, 256), Image.NEAREST), caption=source_label, use_container_width=True)

    pred, proba_dict, feats = predict_image(result_img, bundle)

    with col2:
        st.subheader(f"Predicted Poverty Level: **{pred.upper()}**")
        proba_df = pd.DataFrame(
            {"Class": list(proba_dict.keys()), "Confidence": list(proba_dict.values())}
        ).sort_values("Confidence", ascending=False)
        st.bar_chart(proba_df.set_index("Class"))

    st.subheader("Extracted image features")
    feat_df = pd.DataFrame({"Feature": FEATURE_NAMES, "Value": np.round(feats, 3)})
    st.dataframe(feat_df, hide_index=True, use_container_width=True)
    st.caption(
        "vegetation_ratio & edge_density (roads/building edges) & builtup_ratio "
        "are the strongest signals the model uses — low built-up + high "
        "vegetation typically drives a LOW prediction; dense built-up + high "
        "edge density drives HIGH."
    )
else:
    st.info("Upload an image or generate a synthetic example above to get a prediction.")
