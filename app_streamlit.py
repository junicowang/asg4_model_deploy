"""
ASG 04 – Streamlit App
Spaceship Titanic Model Deployment
"""

import streamlit as st
import pandas as pd
import pickle
from pathlib import Path

BASE_DIR = Path(__file__).parent

# ── Load artifacts ────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    with open(BASE_DIR / "models" / "preprocessor.pkl", "rb") as f:
        obj = pickle.load(f)
    with open(BASE_DIR / "models" / "model.pkl", "rb") as f:
        model = pickle.load(f)
    return obj["preprocessor"], obj["feature_columns"], model

preprocessor, feature_columns, model = load_artifacts()

# ── Page config ───────────────────────────────────────────────
st.set_page_config(page_title="Spaceship Titanic", page_icon="🚀", layout="centered")
st.title("ASG 04 MD - Junico - Spaceship Titanic Model Deployment")
st.markdown("Isi data penumpang di bawah untuk memprediksi apakah mereka **Transported** ke dimensi lain.")
st.divider()

# ── Input form ────────────────────────────────────────────────
st.subheader("🧑 Passenger Information")
col1, col2 = st.columns(2)

with col1:
    passenger_id = st.text_input("PassengerId", value="0001_01")
    home_planet  = st.selectbox("HomePlanet", ["Earth", "Europa", "Mars"], index=0)
    cryo_sleep   = st.selectbox("CryoSleep", [False, True], index=0)
    cabin        = st.text_input("Cabin (Deck/Num/Side)", value="F/123/S")
    destination  = st.selectbox("Destination", ["TRAPPIST-1e", "55 Cancri e", "PSO J318.5-22"], index=0)

with col2:
    age  = st.number_input("Age", min_value=0, max_value=100, value=28)
    vip  = st.selectbox("VIP", [False, True], index=0)
    name = st.text_input("Name", value="John Doe")

st.divider()
st.subheader("💰 Spending (in space credits)")
col3, col4, col5 = st.columns(3)

with col3:
    room_service  = st.number_input("RoomService",  min_value=0.0, value=0.0)
    food_court    = st.number_input("FoodCourt",    min_value=0.0, value=0.0)
with col4:
    shopping_mall = st.number_input("ShoppingMall", min_value=0.0, value=0.0)
    spa           = st.number_input("Spa",          min_value=0.0, value=0.0)
with col5:
    vr_deck       = st.number_input("VRDeck",       min_value=0.0, value=0.0)

st.divider()

# ── Predict button ────────────────────────────────────────────
if st.button("🔮 Predict", use_container_width=True):
    from preprocessing import feature_engineering, transform

    input_df = pd.DataFrame([{
        'PassengerId':  passenger_id,
        'HomePlanet':   home_planet,
        'CryoSleep':    cryo_sleep,
        'Cabin':        cabin,
        'Destination':  destination,
        'Age':          float(age),
        'VIP':          vip,
        'RoomService':  room_service,
        'FoodCourt':    food_court,
        'ShoppingMall': shopping_mall,
        'Spa':          spa,
        'VRDeck':       vr_deck,
        'Name':         name,
    }])

    # Feature engineering + transform
    input_df = feature_engineering(input_df)
    X_input  = transform(input_df, preprocessor, feature_columns)

    # Predict
    prediction  = model.predict(X_input)[0]
    probability = model.predict_proba(X_input)[0][1]

    st.divider()
    if prediction == 1:
        st.success("🌀 **TRANSPORTED!** Penumpang ini dibawa ke dimensi lain.")
    else:
        st.error("🚢 **NOT TRANSPORTED.** Penumpang ini tetap di Spaceship Titanic.")

    st.metric("Transported Probability", f"{probability:.2%}")
    st.progress(float(probability))