"""Streamlit interface for mobile_price_prediction (2).pkl."""

import os
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


MODEL_FILENAME = "mobile_price_prediction (2).pkl"
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

FEATURES = [
    "storage",
    "Battery_cap",
    "RAM",
    "primery_rear_camera",
    "display_size(inch)",
    "refresh_rate(hz)",
]

INPUTS = [
    {"name": "storage", "label": "Storage (GB)", "default": 128.0, "step": 1.0},
    {"name": "Battery_cap", "label": "Battery capacity (mAh)", "default": 5000.0, "step": 100.0},
    {"name": "RAM", "label": "RAM (GB)", "default": 8.0, "step": 1.0},
    {"name": "primery_rear_camera", "label": "Primary rear camera (MP)", "default": 50.0, "step": 1.0},
    {"name": "display_size(inch)", "label": "Display size (inches)", "default": 6.5, "step": 0.1},
    {"name": "refresh_rate(hz)", "label": "Refresh rate (Hz)", "default": 120.0, "step": 1.0},
]


@st.cache_resource
def load_model(model_path: str):
    # The provided model is a joblib file, so use joblib.load rather than pickle.load.
    return joblib.load(model_path)


st.set_page_config(page_title="Mobile Price Prediction", page_icon="📱")
st.title("📱 Mobile Price Prediction")
st.write("Enter the mobile specifications to estimate its price.")

if MODEL_PATH is None:
    st.error(f"Could not find {MODEL_FILENAME}.")
    st.caption(
        "Place the model beside app.py, keep it in C:\\Users\\lenovo\\Downloads, "
        "or set MOBILE_PRICE_MODEL to its full path."
    )
    st.stop()

try:
    model = load_model(str(MODEL_PATH))
except Exception as exc:
    st.error("Could not load the model file.")
    st.exception(exc)
    st.stop()

model_features = getattr(model, "feature_names_in_", None)
if model_features is not None and list(model_features) != FEATURES:
    st.error(
        "The model's input columns do not match this interface. It expects: "
        + ", ".join(map(str, model_features))
    )
    st.stop()

values = {}
with st.form("mobile_specs"):
    columns = st.columns(2)
    for index, item in enumerate(INPUTS):
        with columns[index % 2]:
            values[item["name"]] = st.number_input(
                item["label"],
                min_value=0.0,
                value=item["default"],
                step=item["step"],
            )

    submitted = st.form_submit_button("Predict price", type="primary")

if submitted:
    input_data = pd.DataFrame(
        [[values[name] for name in FEATURES]],
        columns=FEATURES,
    )

    try:
        predicted_price = float(model.predict(input_data)[0])
    except Exception as exc:
        st.error("Prediction failed. Verify that this is the matching model file.")
        st.exception(exc)
        st.stop()

    st.metric("Predicted price", f"{predicted_price:,.2f}")
    st.caption(
        "The result uses the same currency and price unit as the data used to train the model. "
        "Predictions are most reliable within the training data's feature ranges."
    )
