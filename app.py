import streamlit as st
import pandas as pd
import joblib

# --- Page Config ---
st.set_page_config(page_title="Airline Delay Prediction", layout="wide")

# --- Custom CSS ---
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    color: white;
}
.app-header {
    background: linear-gradient(90deg, #000428, #004e92);
    padding: 15px 30px;
    border-radius: 10px;
    text-align: center;
    position: relative;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.6);
}
.ribbon {
    position: absolute;
    top: 0;
    left: 0;
    height: 8px;
    width: 100%;
    background: linear-gradient(90deg, #1e3c72, #2a5298);
    border-radius: 0 0 50% 50%;
    box-shadow: 0px 2px 8px rgba(0,0,0,0.5);
}
.header-title {
    font-size: 32px;
    font-weight: bold;
    color: white;
    margin: 0;
    padding: 10px 0;
    font-family: 'Arial Black', sans-serif;
    letter-spacing: 1.5px;
    vertical-align: middle;
}
.header-logo {
    width: 60px;
    height: 60px;
    vertical-align: middle;
    margin-right: 15px;
}
.form-box {
    background: rgba(255,255,255,0.05);
    padding: 20px;
    border-radius: 15px;
    margin-top: 20px;
}
.form-title {
    font-size: 22px;
    font-weight: bold;
    margin-bottom: 15px;
    text-align: center;
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

# --- Introduction ---
st.markdown("""
Welcome to the **Airline Delay Prediction App**!  
This application predicts **flight delays** based on multiple factors.  

🔹 **You can:**  
- Understand dataset features  
- Enter flight details  
- Predict delay in minutes and class
""", unsafe_allow_html=True)

# --- Load Model, Scaler, Features ---
rf_model = joblib.load("rf_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_columns = joblib.load("feature_columns.pkl")

# --- Day mapping ---
days_map = {"Sunday":1, "Monday":2, "Tuesday":3, "Wednesday":4, "Thursday":5, "Friday":6, "Saturday":7}

# --- Two-column Layout for Inputs ---
col1, col2 = st.columns(2)

with col1:
    carrier_name = st.selectbox("Carrier", ["Delta Air Lines", "United Airlines", "Southwest Airlines", "American Airlines", "JetBlue Airways", "Alaska Airlines", "Spirit Airlines"])
    airport_origin = st.selectbox("Origin Airport", ["ATL - Hartsfield–Jackson Atlanta International", "LAX - Los Angeles International", "ORD - Chicago O'Hare International", "DFW - Dallas/Fort Worth International", "JFK - John F. Kennedy International", "SFO - San Francisco International", "SEA - Seattle–Tacoma International", "MIA - Miami International"])
    airport_dest = st.selectbox("Destination Airport", ["ATL - Hartsfield–Jackson Atlanta International", "LAX - Los Angeles International", "ORD - Chicago O'Hare International", "DFW - Dallas/Fort Worth International", "JFK - John F. Kennedy International", "SFO - San Francisco International", "SEA - Seattle–Tacoma International", "MIA - Miami International"])
    weather_condition = st.selectbox("Weather Condition", ["Clear", "Rain", "Storm", "Fog", "Snow"])
    traffic_level = st.selectbox("Traffic Level", ["Low", "Medium", "High"])

with col2:
    day_of_week = st.selectbox("Day of Week", list(days_map.keys()))
    month = st.selectbox("Month", list(range(1,13)))
    hour = st.slider("Hour of Day", 0, 23, 12)
    distance = st.number_input("Distance (in miles)", min_value=50, max_value=5000, value=500)
    congestion_index = st.slider("Airport Congestion Index", 0, 100, 50)

# --- Predict Button ---
submitted = st.button("🔮 Predict Delay")

# --- Session State ---
if "prediction_made" not in st.session_state:
    st.session_state["prediction_made"] = False

if submitted:
    if airport_origin == airport_dest:
        st.warning("Origin and destination cannot be the same!")
    else:
        input_data = pd.DataFrame({
            "carrier_name": [carrier_name],
            "airport_origin": [airport_origin],
            "airport_dest": [airport_dest],
            "weather_condition": [weather_condition],
            "traffic_level": [traffic_level],
            "day_of_week": [days_map[day_of_week]],
            "month": [month],
            "hour": [hour],
            "distance": [distance],
            "airport_congestion_index": [congestion_index]
        })

        input_encoded = pd.get_dummies(input_data)
        input_encoded = input_encoded.reindex(columns=feature_columns, fill_value=0)

        input_scaled = scaler.transform(input_encoded)
        predicted_delay = rf_model.predict(input_scaled)[0]
        predicted_class = "On-Time ✈️" if predicted_delay <= 15 else "Delayed ⏱️"

        st.session_state["prediction_made"] = True
        st.session_state["predicted_delay"] = round(predicted_delay, 2)
        st.session_state["predicted_class"] = predicted_class

# --- Show Results ---
if st.session_state["prediction_made"]:
    st.markdown(f"""
        <div style='margin-top:20px; padding:25px; border-radius:15px; background: linear-gradient(90deg, #000428, #004e92); color:white; text-align:center; box-shadow:0px 6px 20px rgba(0,0,0,0.7);'>
            <h2 style='margin-bottom:15px;'>🔮 Prediction Results</h2>
            <h3>Predicted Arrival Delay: <span style='color:#00ffcc;'>{st.session_state['predicted_delay']} minutes</span></h3>
            <h3>Status: <span style='color:#ffd700;'>{st.session_state['predicted_class']}</span></h3>
        </div>
    """, unsafe_allow_html=True)
