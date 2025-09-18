import streamlit as st

# --- Page Config ---
st.set_page_config(
    page_title="Airline Delay Prediction",
    layout="wide"
)

# --- Custom CSS for Modern Gradient UI ---
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        color: white;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }

    /* Header Container */
    .app-header {
        background: linear-gradient(90deg, #000428, #004e92);
        padding: 25px 40px;
        border-radius: 15px;
        text-align: center;
        position: relative;
        box-shadow: 0px 6px 20px rgba(0,0,0,0.7);
    }

    /* Animated Ribbon */
    .ribbon {
        position: absolute;
        top: 0;
        left: 0;
        height: 10px;
        width: 100%;
        background: linear-gradient(90deg, #1e3c72, #2a5298, #1e3c72);
        border-radius: 0 0 50% 50%;
        animation: slide 6s linear infinite;
    }
    @keyframes slide {
        0% { background-position: 0% 50%; }
        100% { background-position: 100% 50%; }
    }

    /* Title */
    .header-title {
        font-size: 40px;
        font-weight: bold;
        color: #ffffff;
        margin: 0;
        padding: 10px 0;
        font-family: 'Arial Black', sans-serif;
        letter-spacing: 2px;
        text-shadow: 2px 2px 8px rgba(0,0,0,0.8);
    }

    /* Logo */
    .header-logo {
        width: 70px;
        height: 70px;
        vertical-align: middle;
        margin-right: 20px;
    }

    /* Intro Box */
    .intro-box {
        margin-top: 30px;
        background: rgba(0,0,0,0.5);
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0px 4px 15px rgba(0,0,0,0.5);
        font-size: 18px;
        line-height: 1.6;
    }
    </style>
""", unsafe_allow_html=True)

# --- Header Section ---
st.markdown("""
    <div class="app-header">
        <div class="ribbon"></div>
        <img class="header-logo" src="https://img.icons8.com/ios/100/ffffff/airport.png" alt="Airplane logo">
        <span class="header-title">Airline Delay Prediction App</span>
    </div>
""", unsafe_allow_html=True)

# --- Introduction Section ---
st.markdown("""
    <div class="intro-box">
        ✈Welcome to the <b>Airline Delay Prediction App</b>! <br><br>
        This application helps you <b>predict flight delays</b> based on multiple real-world factors 
        such as <b>weather, traffic, distance, congestion</b> and more.  
        <br><br>
        🌐 With just a few flight details, you can:  
        - Explore the dataset  
        - Understand flight delay patterns  
        - Enter custom flight information  
        - Get predictions for <b>arrival delay (minutes)</b> and <b>delay class</b>  
        <br>
    </div>
""", unsafe_allow_html=True)

import joblib
import pandas as pd

# --- Load model, scaler, and feature columns ---
rf_model = joblib.load("rf_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_columns = joblib.load("feature_columns.pkl")

# --- Input + Prediction Section ---
st.markdown('<div class="form-box">', unsafe_allow_html=True)
st.markdown('<div class="form-title">✈ Enter Flight Details</div>', unsafe_allow_html=True)

# Day mapping (UI shows names, model gets numbers)
days_map = {
    "Sunday": 1, "Monday": 2, "Tuesday": 3,
    "Wednesday": 4, "Thursday": 5,
    "Friday": 6, "Saturday": 7
}

# Two-column layout
col1, col2 = st.columns(2)

with col1:
    carrier_name = st.selectbox("Carrier", 
                                ["Delta Air Lines", "United Airlines", "Southwest Airlines", 
                                 "American Airlines", "JetBlue Airways", "Alaska Airlines", "Spirit Airlines"])
    airport_origin = st.selectbox("Origin Airport", 
                                  ["ATL - Hartsfield–Jackson Atlanta International", 
                                   "LAX - Los Angeles International", 
                                   "ORD - Chicago O'Hare International", 
                                   "DFW - Dallas/Fort Worth International", 
                                   "JFK - John F. Kennedy International", 
                                   "SFO - San Francisco International", 
                                   "SEA - Seattle–Tacoma International", 
                                   "MIA - Miami International"])
    airport_dest = st.selectbox("Destination Airport", 
                                ["ATL - Hartsfield–Jackson Atlanta International", 
                                 "LAX - Los Angeles International", 
                                 "ORD - Chicago O'Hare International", 
                                 "DFW - Dallas/Fort Worth International", 
                                 "JFK - John F. Kennedy International", 
                                 "SFO - San Francisco International", 
                                 "SEA - Seattle–Tacoma International", 
                                 "MIA - Miami International"])
    weather_condition = st.selectbox("Weather Condition", 
                                     ["Clear", "Rain", "Storm", "Fog", "Snow"])
    traffic_level = st.selectbox("Traffic Level", ["Low", "Medium", "High"])

with col2:
    day_of_week = st.selectbox("Day of Week", list(days_map.keys()))
    month = st.selectbox("Month", list(range(1,13)))
    hour = st.slider("Hour of Day", 0, 23, 12)
    distance = st.number_input("Distance (in miles)", min_value=50, max_value=5000, value=500)
    congestion_index = st.slider("Airport Congestion Index", 0, 100, 50)

# Predict button
submitted = st.button("🔮 Predict Delay")

# --- Session State Initialization ---
if "prediction_made" not in st.session_state:
    st.session_state["prediction_made"] = False
if "predicted_delay" not in st.session_state:
    st.session_state["predicted_delay"] = None
if "predicted_class" not in st.session_state:
    st.session_state["predicted_class"] = None

# --- On Submit: Process + Predict ---
if submitted:
    # Build input DataFrame
    input_data = pd.DataFrame({
        "carrier_name": [carrier_name],
        "airport_origin": [airport_origin],
        "airport_dest": [airport_dest],
        "weather_condition": [weather_condition],
        "traffic_level": [traffic_level],
        "day_of_week": [days_map[day_of_week]],  # mapped number
        "month": [month],
        "hour": [hour],
        "distance": [distance],
        "airport_congestion_index": [congestion_index]
    })

    # Encode + align
    input_encoded = pd.get_dummies(input_data)
    input_encoded = input_encoded.reindex(columns=feature_columns, fill_value=0)

    # Scale
    input_scaled = scaler.transform(input_encoded)

    # Predict
    predicted_delay = rf_model.predict(input_scaled)[0]
    predicted_class = "On-Time ✈️" if predicted_delay <= 15 else "Delayed ⏱️"

    # Save to session
    st.session_state["prediction_made"] = True
    st.session_state["predicted_delay"] = round(predicted_delay, 2)
    st.session_state["predicted_class"] = predicted_class

# --- Show Results ---
if st.session_state["prediction_made"]:
    st.markdown("""
        <div style="margin-top:20px; padding:25px; border-radius:15px; 
                    background: linear-gradient(90deg, #000428, #004e92);
                    color:white; text-align:center; 
                    box-shadow:0px 6px 20px rgba(0,0,0,0.7);">
            <h2 style="margin-bottom:15px;">🔮 Prediction Results</h2>
            <h3>Predicted Arrival Delay: <span style="color:#00ffcc;">{} minutes</span></h3>
            <h3>Status: <span style="color:#ffd700;">{}</span></h3>
        </div>
    """.format(st.session_state["predicted_delay"], 
               st.session_state["predicted_class"]), unsafe_allow_html=True)
