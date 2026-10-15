"""Streamlit interface for the mobile price regression model in the notebook."""

import os
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


# Defaults to the model saved from the user's notebook. Override this with the
# MOBILE_PRICE_MODEL environment variable if the file is in another location.
MODEL_PATH = Path(
    os.environ.get(
        "MOBILE_PRICE_MODEL",
        r"mobile_price_prediction.pkl",
    )
)


@st.cache_resource
def load_model(model_path: str):
    """Load the fitted LinearRegression model once per Streamlit session."""
    return joblib.load(model_path)


st.set_page_config(page_title="Mobile Price Prediction", page_icon="📱")
st.title("📱 Mobile Price Prediction")
st.write(
    "Enter the phone's storage and RAM. The fitted regression model will "
    "estimate its price."
)

if not MODEL_PATH.is_file():
    st.error(f"Model file not found: {MODEL_PATH}")
    st.caption(
        "Save mobile_price_prediction.pkl at that location, or set the "
        "MOBILE_PRICE_MODEL environment variable to its full path."
    )
    st.stop()

try:
    model = load_model(str(MODEL_PATH))
except Exception as exc:
    st.error("Could not load the model file.")
    st.exception(exc)
    st.stop()

with st.form("mobile_specs"):
    storage_col, ram_col = st.columns(2)
    with storage_col:
        storage = st.number_input(
            "Storage",
            min_value=0.0,
            value=256.0,
            step=1.0,
            help="Enter the storage value using the same unit as the training data.",
        )
    with ram_col:
        ram = st.number_input(
            "RAM",
            min_value=0.0,
            value=64.0,
            step=1.0,
            help="Enter the RAM value using the same unit as the training data.",
        )

    submitted = st.form_submit_button("Predict price", type="primary")

if submitted:
    # Match the notebook's feature names and order exactly: x = df[['storage', 'RAM']].
    input_data = pd.DataFrame({"storage": [storage], "RAM": [ram]})

    try:
        predicted_price = float(model.predict(input_data)[0])
    except Exception as exc:
        st.error(
            "Prediction failed. Confirm that the selected model was trained "
            "with the `storage` and `RAM` columns, in that order."
        )
        st.exception(exc)
        st.stop()

    st.metric("Predicted price", f"{predicted_price:,.2f}")
    st.caption(
        "The result uses the same unit and currency as the notebook's `Price` column. "
        "Predictions are most reliable within the storage and RAM values used to train the model."
    )
