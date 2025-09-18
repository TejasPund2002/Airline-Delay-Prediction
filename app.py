import streamlit as st
import joblib
import pandas as pd

# --- Page Configuration ---
st.set_page_config(
    page_title="Flight Delay Predictor",
    page_icon="✈️",
    layout="centered"
)

# --- Custom CSS for Styling ---
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #f0f4f8, #c8d3e6);
        color: #1a1a1a;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .stSelectbox, .stNumberInput, .stSlider {
        margin-bottom: 1rem;
    }
    .form-box {
        background-color: #ffffff;
        padding: 30px;
        border-radius: 15px;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.1);
        border: 1px solid #e0e0e0;
        margin-bottom: 2rem;
    }
    .form-title {
        font-size: 2rem;
        font-weight: bold;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .stButton > button {
        background-color: #3498db;
        color: white;
        font-weight: bold;
        padding: 0.75rem 2rem;
        border-radius: 10px;
        border: none;
        transition: transform 0.2s, background-color 0.2s;
    }
    .stButton > button:hover {
        background-color: #2980b9;
        transform: scale(1.05);
    }
    .result-box {
        margin-top: 20px;
        padding: 25px;
        border-radius: 15px;
        background: linear-gradient(90deg, #000428, #004e92);
        color: white;
        text-align: center;
        box-shadow: 0 6px 20px rgba(0,0,0,0.7);
    }
    .result-box h2 {
        margin-bottom: 15px;
    }
    .result-box h3 {
        margin-bottom: 10px;
    }
    .highlight {
        color: #00ffcc;
        font-weight: bold;
    }
    .status-highlight {
        color: #ffd700;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


# --- Load model, scaler, and feature columns with caching ---
@st.cache_resource
def load_resources():
    """Loads the model, scaler, and feature columns once."""
    try:
        rf_model = joblib.load("rf_model.pkl")
        scaler = joblib.load("scaler.pkl")
        feature_columns = joblib.load("feature_columns.pkl")
        return rf_model, scaler, feature_columns
    except FileNotFoundError:
        st.error("Model files not found. Please ensure 'rf_model.pkl', 'scaler.pkl', and 'feature_columns.pkl' are in the same directory as the app.")
        st.stop()
    except Exception as e:
        st.error(f"An error occurred loading the model files: {e}")
        st.stop()

rf_model, scaler, feature_columns = load_resources()

# --- Input + Prediction Section ---
st.markdown('<div class="form-box">', unsafe_allow_html=True)
st.markdown('<h1 class="form-title">✈️ Flight Delay Predictor</h1>', unsafe_allow_html=True)

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
    month = st.selectbox("Month", list(range(1, 13)))
    hour = st.slider("Hour of Day", 0, 23, 12)
    distance = st.number_input("Distance (in miles)", min_value=50, max_value=5000, value=500)
    congestion_index = st.slider("Airport Congestion Index", 0, 100, 50)

# Predict button
submitted = st.button("🔮 Predict Delay")

st.markdown('</div>', unsafe_allow_html=True)

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

    # --- Show Results ---
    st.markdown(f"""
        <div class="result-box">
            <h2>🔮 Prediction Results</h2>
            <h3>Predicted Arrival Delay: <span class="highlight">{round(predicted_delay, 2)} minutes</span></h3>
            <h3>Status: <span class="status-highlight">{predicted_class}</span></h3>
        </div>
    """, unsafe_allow_html=True)
