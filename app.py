"""Streamlit interface for Bluetooth signal-quality prediction."""
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "bluetooth_quality_model.joblib"
DATA_PATH = ROOT / "data" / "bluetooth_signal_dataset.csv"
st.set_page_config(page_title="Bluetooth Signal Quality AI", page_icon="📶", layout="wide")
st.title("📶 AI-Based Bluetooth Signal Quality Analysis")
st.caption("Capstone Project 14 • Educational prototype • Included dataset is synthetic demonstration data")

if not MODEL_PATH.exists():
    st.warning("The trained model was not found. Run `python train_model.py` in the project folder, then refresh this page.")
    st.stop()

model = joblib.load(MODEL_PATH)
left, right = st.columns([1, 1])
with left:
    st.subheader("Enter Bluetooth link conditions")
    distance = st.slider("Distance (metres)", 1.0, 30.0, 5.0, 0.5)
    rssi = st.slider("RSSI (dBm)", -100.0, -25.0, -55.0, 1.0)
    packet_loss = st.slider("Packet loss (%)", 0.0, 100.0, 2.0, 0.5)
    latency = st.slider("Latency (ms)", 2.0, 300.0, 25.0, 1.0)
    interference = st.selectbox("Interference", ["low", "medium", "high"])
    obstacles = st.slider("Obstacles", 0, 4, 0)
    status = st.selectbox("Connection status", ["connected", "unstable"])
    if st.button("Predict quality", type="primary", use_container_width=True):
        row = pd.DataFrame([{
            "distance_m": distance, "rssi_dbm": rssi,
            "packet_loss_pct": packet_loss, "latency_ms": latency,
            "interference": interference, "obstacles": obstacles,
            "connection_status": status
        }])
        pred = model.predict(row)[0]
        probs = model.predict_proba(row)[0]
        classes = model.named_steps["classifier"].classes_
        st.session_state["prediction"] = pred
        st.session_state["probabilities"] = dict(zip(classes, probs))

with right:
    st.subheader("Prediction")
    if "prediction" in st.session_state:
        pred = st.session_state["prediction"]
        if pred == "good":
            st.success(f"Predicted link quality: {pred.upper()}")
        elif pred == "moderate":
            st.warning(f"Predicted link quality: {pred.upper()}")
        else:
            st.error(f"Predicted link quality: {pred.upper()}")
        st.write("Model class probabilities (not a guarantee):")
        st.bar_chart(pd.Series(st.session_state["probabilities"], name="Probability"))
    else:
        st.info("Set the conditions on the left and click **Predict quality**.")
    st.markdown("""
    **Interpretation guide**
    - **Good:** comparatively strong/stable link conditions.
    - **Moderate:** usable link with some degradation.
    - **Poor:** weak or unreliable link conditions.

    These labels are learned from the included demo dataset. Replace it with real, consistently labelled measurements for valid field results.
    """)

st.divider()
st.subheader("Explore the sample dataset")
if DATA_PATH.exists():
    data = pd.read_csv(DATA_PATH)
    c1, c2, c3 = st.columns(3)
    c1.metric("Demo samples", f"{len(data):,}")
    c2.metric("Features", "7")
    c3.metric("Quality classes", data["quality"].nunique())
    st.bar_chart(data.groupby("quality")[["rssi_dbm", "packet_loss_pct", "latency_ms"]].mean())
    with st.expander("View sample rows"):
        st.dataframe(data.head(30), use_container_width=True)
else:
    st.info("Run `python generate_dataset.py` to create the sample dataset.")
