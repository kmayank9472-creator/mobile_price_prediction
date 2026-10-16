"""Streamlit interface for mobile_price_prediction.pkl."""

import os
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


MODEL_FILENAME = "mobile_price_prediction.pkl"
MODEL_CANDIDATES = []
if os.environ.get("MOBILE_PRICE_MODEL"):
    MODEL_CANDIDATES.append(Path(os.environ["MOBILE_PRICE_MODEL"]))
MODEL_CANDIDATES.extend(
    [
        Path(__file__).resolve().with_name(MODEL_FILENAME),
        Path(r"C:\Users\lenovo\Downloads") / MODEL_FILENAME,
    ]
)
MODEL_PATH = next((path for path in MODEL_CANDIDATES if path.is_file()), None)


@st.cache_resource
def load_model(model_path: str):
    # This model was saved with joblib.dump(), so load it with joblib.load().
    return joblib.load(model_path)


st.set_page_config(page_title="Mobile Price Prediction", page_icon="📱")
st.title("📱 Mobile Price Prediction")
st.write("Enter the mobile's storage and battery capacity to estimate its price.")

if MODEL_PATH is None:
    st.error(f"Could not find {MODEL_FILENAME}.")
    st.caption(
        "Put the model beside app.py, keep it in C:\\Users\\lenovo\\Downloads, "
        "or set MOBILE_PRICE_MODEL to its full file path."
    )
    st.stop()

try:
    model = load_model(str(MODEL_PATH))
except Exception as exc:
    st.error("Could not load the model. It must be the joblib model file.")
    st.exception(exc)
    st.stop()

expected_features = ["storage", "Battery_cap"]
model_features = getattr(model, "feature_names_in_", None)
if model_features is not None and list(model_features) != expected_features:
    st.error(
        "This model expects different input columns: "
        + ", ".join(map(str, model_features))
    )
    st.stop()

with st.form("mobile_specs"):
    storage_col, battery_col = st.columns(2)
    with storage_col:
        storage = st.number_input(
            "Storage",
            min_value=0.0,
            value=128.0,
            step=1.0,
            help="Use the same unit used in the model's training data.",
        )
    with battery_col:
        battery_cap = st.number_input(
            "Battery capacity",
            min_value=0.0,
            value=5000.0,
            step=100.0,
            help="Use the same unit used in the model's training data (often mAh).",
        )

    submitted = st.form_submit_button("Predict price", type="primary")

if submitted:
    # Keep both names and order aligned with feature_names_in_ from the model.
    input_data = pd.DataFrame(
        {"storage": [storage], "Battery_cap": [battery_cap]}
    )

    try:
        predicted_price = float(model.predict(input_data)[0])
    except Exception as exc:
        st.error(
            "Prediction failed. Check that this model was trained with "
            "`storage` and `Battery_cap`."
        )
        st.exception(exc)
        st.stop()

    st.metric("Predicted price", f"{predicted_price:,.2f}")
    st.caption(
        "The result uses the same unit and currency as the training data's Price column. "
        "Predictions are most reliable within the training data's feature ranges."
    )
