import os
import base64
import pandas as pd
import requests
import streamlit as st

# Base URL of the FastAPI backend (can be overridden with the API_URL env variable)
API_BASE_URL = os.getenv("API_URL", "http://localhost:8000")
PREDICT_URL = f"{API_BASE_URL}/predict"
MLFLOW_URL = "http://127.0.0.1:5001"
IMAGES_DIR = os.path.join(os.path.dirname(__file__), "images")


def show_image(filename: str, caption: str | None = None) -> None:
    """Embed an image from the images folder directly into the page (base64)."""
    path = os.path.join(IMAGES_DIR, filename)
    if not os.path.exists(path):
        return

    with open(path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()

    ext = os.path.splitext(filename)[1].lstrip(".").lower()
    mime = "jpeg" if ext in ("jpg", "jpeg") else ext

    st.markdown(
        f'<img src="data:image/{mime};base64,{encoded}" '
        f'style="width:100%; border-radius:12px; margin-bottom:8px;">',
        unsafe_allow_html=True,
    )
    if caption:
        st.caption(caption)

st.set_page_config(page_title="Wine Classifier", page_icon="🍷", layout="centered")

# Features in the same order as WineFeatures in main.py:
# (API field name, label shown to the user, typical min, typical max in the dataset)
FEATURES = [
    ("alcohol", "Alcohol", 11.03, 14.83),
    ("malic_acid", "Malic acid", 0.74, 5.80),
    ("ash", "Ash", 1.36, 3.23),
    ("alcalinity_of_ash", "Alcalinity of ash", 10.6, 30.0),
    ("magnesium", "Magnesium", 70.0, 162.0),
    ("total_phenols", "Total phenols", 0.98, 3.88),
    ("flavanoids", "Flavanoids", 0.34, 5.08),
    ("nonflavanoid_phenols", "Nonflavanoid phenols", 0.13, 0.66),
    ("proanthocyanins", "Proanthocyanins", 0.41, 3.58),
    ("color_intensity", "Color intensity", 1.28, 13.0),
    ("hue", "Hue", 0.48, 1.71),
    ("od280_od315_of_diluted_wines", "OD280/OD315 of diluted wines", 1.27, 4.0),
    ("proline", "Proline", 278.0, 1680.0),
]

# Real samples from the Wine dataset, one for each class
EXAMPLES = {
    "Example — class 0": [14.23, 1.71, 2.43, 15.6, 127, 2.80, 3.06, 0.28, 2.29, 5.64, 1.04, 3.92, 1065],
    "Example — class 1": [12.37, 0.94, 1.36, 10.6, 88, 1.98, 0.57, 0.28, 0.42, 1.95, 1.05, 1.82, 520],
    "Example — class 2": [12.86, 1.35, 2.32, 18.0, 122, 1.51, 1.25, 0.21, 0.94, 4.10, 0.76, 1.29, 630],
}

# The three classes are three grape cultivars from the same region of Italy
CLASS_INFO = {0: "Cultivar 1", 1: "Cultivar 2", 2: "Cultivar 3"}


def api_is_online() -> bool:
    """Check the health endpoint of the FastAPI backend."""
    try:
        return requests.get(API_BASE_URL, timeout=2).ok
    except requests.exceptions.RequestException:
        return False


# ---------- Sidebar ----------
with st.sidebar:
    show_image("banner.jpg")
    st.header("About")
    st.write(
        "This app predicts the class of a wine sample using a Random Forest "
        "model trained on the scikit-learn Wine dataset (178 samples, 13 features, 3 classes)."
    )

    st.subheader("Service status")
    if api_is_online():
        st.success("FastAPI backend is online")
    else:
        st.error("FastAPI backend is offline")

    st.subheader("Links")
    st.markdown(f"- [FastAPI docs]({API_BASE_URL}/docs)\n- [MLflow UI]({MLFLOW_URL})")

# ---------- Main page ----------
st.title("🍷 Wine Classification App")
show_image("banner.jpg")
st.caption("Streamlit frontend → FastAPI backend → Random Forest model (tracked with MLflow)")

# Choosing an example reloads the inputs with that sample's values
example = st.selectbox("Load an example sample", list(EXAMPLES))
default_values = EXAMPLES[example]

# A form sends the request only when the button is pressed
with st.form("features_form"):
    st.subheader("Wine features")
    cols = st.columns(2)
    payload = {}
    for i, ((key, label, low, high), default) in enumerate(zip(FEATURES, default_values)):
        payload[key] = cols[i % 2].number_input(
            label,
            value=float(default),
            key=f"{key}_{example}",  # new key per example so the input resets
            help=f"Typical range in the dataset: {low} – {high}",
        )
    submitted = st.form_submit_button("Predict", type="primary")

# ---------- Prediction ----------
if submitted:
    try:
        with st.spinner("Asking the model..."):
            response = requests.post(PREDICT_URL, json=payload, timeout=5)
        response.raise_for_status()
        result = response.json()
        prediction = result["prediction"]

        st.success("Prediction received!")
        c1, c2 = st.columns(2)
        c1.metric("Predicted class", result["class_name"])
        c2.metric("Grape cultivar", CLASS_INFO.get(prediction, "Unknown"))
        show_image(f"class_{prediction}.jpg", caption=CLASS_INFO.get(prediction))

        with st.expander("Show details"):
            st.write("Input sent to the API:")
            st.dataframe(pd.DataFrame.from_dict(payload, orient="index", columns=["value"]))
            st.write("Raw API response:")
            st.json(result)

    except requests.exceptions.ConnectionError:
        st.error(f"Cannot connect to the API at {API_BASE_URL}. Make sure FastAPI is running.")
    except requests.exceptions.Timeout:
        st.error("The API took too long to respond. Try again.")
    except requests.exceptions.HTTPError:
        st.error(f"API error {response.status_code}: {response.text}")