"""
Streamlit deployment app for the Mobile Price Prediction model (Group 10).

Run with:
    streamlit run app.py

Expects "mobile_price_model.joblib" (produced by Group-10.ipynb, Section 7.1)
to be present in the same directory. That artifact bundles:
    - model         : the fitted Voting Ensemble (RF + SVM + GB)
    - scaler        : the StandardScaler fitted on the training data
    - feature_cols  : exact column order the model was trained on
    - binary_map    : {"Yes": 1, "No": 0}
    - ordinal_map   : {"Low": 0, "Med": 1, "High": 2}
    - class_labels  : {0: "Low cost", 1: "Medium cost", 2: "High cost", 3: "Very high cost"}
"""

import pandas as pd
import streamlit as st
import joblib

MODEL_PATH = "mobile_price_model.joblib"

st.set_page_config(page_title="Mobile Price Range Predictor", page_icon="📱", layout="centered")


@st.cache_resource
def load_artifacts():
    return joblib.load(MODEL_PATH)


def predict_price_range(art, raw_features: dict):
    """Encode raw (Yes/No, Low/Med/High) inputs exactly as in training, then predict."""
    row = pd.DataFrame([raw_features])
    for col in ["blue", "dual_sim", "touch_screen", "wifi"]:
        row[col] = row[col].map(art["binary_map"])
    row["mobile_wt"] = row["mobile_wt"].map(art["ordinal_map"])
    row = row[art["feature_cols"]]          # enforce training column order
    x = art["scaler"].transform(row)        # identical transformation as training
    pred = int(art["model"].predict(x)[0])
    proba = None
    if hasattr(art["model"], "predict_proba"):
        proba = art["model"].predict_proba(x)[0]
    return pred, proba


st.title("📱 Mobile Price Range Predictor")
st.caption(
    "Group 10 — Introduction to Data Science (S2-25_DSECLZG532). "
    "Predicts the price band (Low / Medium / High / Very high) of a phone from its specs, "
    "using a soft-voting ensemble (Random Forest + SVM + Gradient Boosting)."
)

try:
    artifacts = load_artifacts()
except FileNotFoundError:
    st.error(
        f"Model file '{MODEL_PATH}' not found. Please run Group-10.ipynb "
        "(Section 7.1) first to train and save the model artifacts."
    )
    st.stop()

st.sidebar.header("Phone Specifications")

# Numeric inputs -- most influential features (per Section 4.6) shown first for prominence
st.sidebar.subheader("Key specs")
ram = st.sidebar.slider("RAM (MB)", 256, 3998, 2000, step=2)
battery_power = st.sidebar.slider("Battery power (mAh)", 500, 2000, 1200, step=1)
px_height = st.sidebar.slider("Pixel resolution height", 0, 1960, 600, step=1)
px_width = st.sidebar.slider("Pixel resolution width", 500, 2000, 1200, step=1)

st.sidebar.subheader("Other specs")
clock_speed = st.sidebar.slider("Clock speed (GHz)", 0.5, 3.0, 1.5, step=0.1)
fc = st.sidebar.slider("Front camera (MP)", 0, 19, 4)
pc = st.sidebar.slider("Primary camera (MP)", 0, 20, 10)
int_memory = st.sidebar.slider("Internal memory (GB)", 2, 64, 32)
m_dep = st.sidebar.slider("Mobile depth (cm)", 0.1, 1.0, 0.5, step=0.1)
n_cores = st.sidebar.slider("Number of cores", 1, 8, 4)
sc_h = st.sidebar.slider("Screen height (cm)", 5, 19, 12)
sc_w = st.sidebar.slider("Screen width (cm)", 1, 18, 6)
talk_time = st.sidebar.slider("Talk time (hours)", 2, 20, 11)

st.sidebar.subheader("Features / connectivity")
mobile_wt = st.sidebar.selectbox("Mobile weight category", ["Low", "Med", "High"], index=1)
blue = st.sidebar.selectbox("Bluetooth", ["Yes", "No"], index=1)
dual_sim = st.sidebar.selectbox("Dual SIM", ["Yes", "No"], index=1)
four_g = st.sidebar.selectbox("4G", ["Yes", "No"], index=1)
three_g = st.sidebar.selectbox("3G", ["Yes", "No"], index=0)
touch_screen = st.sidebar.selectbox("Touch screen", ["Yes", "No"], index=0)
wifi = st.sidebar.selectbox("WiFi", ["Yes", "No"], index=1)

raw_features = {
    "battery_power": battery_power,
    "blue": blue,
    "clock_speed": clock_speed,
    "dual_sim": dual_sim,
    "fc": fc,
    "four_g": 1 if four_g == "Yes" else 0,
    "int_memory": int_memory,
    "m_dep": m_dep,
    "mobile_wt": mobile_wt,
    "n_cores": n_cores,
    "pc": pc,
    "px_height": px_height,
    "px_width": px_width,
    "ram": ram,
    "sc_h": sc_h,
    "sc_w": sc_w,
    "talk_time": talk_time,
    "three_g": 1 if three_g == "Yes" else 0,
    "touch_screen": touch_screen,
    "wifi": wifi,
}

st.subheader("Selected specifications")
st.dataframe(pd.DataFrame([raw_features]), use_container_width=True)

if st.button("Predict price range", type="primary"):
    pred, proba = predict_price_range(artifacts, raw_features)
    label = artifacts["class_labels"][pred]

    st.success(f"Predicted price_range = **{pred}** ({label})")

    if proba is not None:
        proba_df = pd.DataFrame(
            {"price_range": list(artifacts["class_labels"].values()), "probability": proba}
        ).set_index("price_range")
        st.subheader("Class probabilities")
        st.bar_chart(proba_df)

st.markdown("---")
st.caption(
    "Model: soft-voting ensemble (Random Forest + RBF-SVM + Gradient Boosting), "
    "93.75% validation accuracy / macro-F1 ≈ 0.94, 5-fold CV 91.4% ± 1.8% "
    "(see Group-10.ipynb, Sections 5-6)."
)
