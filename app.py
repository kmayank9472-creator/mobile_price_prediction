import os
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


MODEL_PATH = Path(
    os.environ.get(
        "MOBILE_PRICE_MODEL",
        r"C:\Users\lenovo\Downloads\mobile_price_prediction.pkl",
    )
)


@st.cache_resource
def load_model(model_path):
    return joblib.load(model_path)


st.set_page_config(page_title="Mobile Price Prediction", page_icon="📱")
st.title("📱 Mobile Price Prediction")
st.write("Phone ka storage aur RAM enter karo, phir price predict karo.")

if not MODEL_PATH.is_file():
    st.error(f"Model file nahi mili: {MODEL_PATH}")
    st.stop()

try:
    model = load_model(str(MODEL_PATH))
except Exception as exc:
    st.error("Model file load nahi hui.")
    st.exception(exc)
    st.stop()

with st.form("mobile_specs"):
    col1, col2 = st.columns(2)

    with col1:
        storage = st.number_input(
            "Storage",
            min_value=0.0,
            value=256.0,
            step=1.0,
        )

    with col2:
        ram = st.number_input(
            "RAM",
            min_value=0.0,
            value=64.0,
            step=1.0,
        )

    submitted = st.form_submit_button("Predict price")

if submitted:
    # Notebook ke feature names aur order se match karta hai
    input_data = pd.DataFrame({
        "storage": [storage],
        "RAM": [ram],
    })

    try:
        predicted_price = float(model.predict(input_data)[0])
        st.metric("Predicted price", f"{predicted_price:,.2f}")
        st.caption("Result ka unit/currency notebook ke Price column jaisa hoga.")
    except Exception as exc:
        st.error("Prediction nahi ho paya. Model ko `storage` aur `RAM` features par train kiya hona chahiye.")
        st.exception(exc)
