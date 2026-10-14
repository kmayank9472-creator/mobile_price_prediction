import streamlit as st
import numpy as np
import pandas as pd
import joblib
from pathlib import Path


st.set_page_config(page_title="Mobile Price Predictor", page_icon="📱")

MODEL_PATH = Path("mobile_price_model.joblib")

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

st.title("📱 Mobile Price Prediction")
st.write("Phone ki specifications enter karke estimated price dekhein.")

if not MODEL_PATH.exists():
    st.error(f"Model file nahi mili: {MODEL_PATH}")
    st.info("Trained model ko `mobile_price_model.joblib` naam se app.py ke folder mein rakhein.")
    st.stop()

try:
    model = load_model()
except Exception as error:
    st.error(f"Model load nahi ho saka: {error}")
    st.stop()

# Model ke feature names available hon to unhe use karein.
feature_names = getattr(model, "feature_names_in_", None)
if feature_names is None:
    feature_names = [
        "ram_gb",
        "storage_gb",
        "battery_mah",
        "camera_mp",
        "screen_inches",
    ]

st.subheader("Phone specifications")

col1, col2 = st.columns(2)

with col1:
    ram = st.number_input("RAM (GB)", min_value=1, max_value=32, value=8)
    storage = st.number_input("Storage (GB)", min_value=8, max_value=2048, value=128)
    battery = st.number_input("Battery (mAh)", min_value=1000, max_value=10000, value=5000)

with col2:
    camera = st.number_input("Main camera (MP)", min_value=2, max_value=250, value=50)
    screen = st.number_input(
        "Screen size (inches)", min_value=4.0, max_value=8.0, value=6.5, step=0.1
    )

if st.button("Price predict karein", type="primary"):
    # Input names aur training data ke column names match hone chahiye.
    values = {
        "ram_gb": ram,
        "storage_gb": storage,
        "battery_mah": battery,
        "camera_mp": camera,
        "screen_inches": screen,
    }

    try:
        input_df = pd.DataFrame(
            [[values[name] for name in feature_names]],
            columns=feature_names,
        )
        prediction = model.predict(input_df)[0]

        # Numeric output ko rupees mein dikhayein; text/category ko seedha dikhayein.
        if isinstance(prediction, (int, float, np.integer, np.floating)):
            st.success(f"Estimated mobile price: ₹{prediction:,.0f}")
        else:
            st.success(f"Predicted price category: {prediction}")

    except KeyError as error:
        st.error(f"Model ka feature app ke inputs se match nahi hua: {error}")
    except Exception as error:
        st.error(f"Prediction nahi ho saki: {error}")
