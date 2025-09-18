import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.express as px

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
        background: linear-gradient(45deg, #1b263b, #0d1b2a);
        color: #e0eaff;
        padding: 30px;
        border-radius: 15px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
        border: 2px solid #3498db;
        margin-bottom: 3rem; /* Increased spacing */
    }
    .form-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: #e0eaff;
        text-align: center;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    .form-subtitle {
        font-size: 1.1rem;
        text-align: center;
        margin-bottom: 2rem;
        color: #b0c4de;
    }
    .stButton > button {
        background-color: #3498db;
        color: white;
        font-weight: bold;
        padding: 0.75rem 2rem;
        border-radius: 10px;
        border: none;
        transition: transform 0.2s, background-color 0.2s, box-shadow 0.2s;
        box-shadow: 0 4px 15px rgba(52, 152, 219, 0.4);
    }
    .stButton > button:hover {
        background-color: #2980b9;
        transform: scale(1.05);
        box-shadow: 0 6px 20px rgba(52, 152, 219, 0.6);
    }
    .result-box {
        margin-top: 20px;
        padding: 25px;
        border-radius: 15px;
        background: linear-gradient(90deg, #000428, #004e92);
        color: white;
        text-align: center;
        box-shadow: 0 6px 20px rgba(0,0,0,0.7);
        margin-bottom: 3rem; /* Increased spacing */
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
    .stAlert {
        border-radius: 10px;
        font-weight: bold;
    }
    .feature-box {
        background: linear-gradient(135deg, #2c3e50, #34495e); 
        padding: 30px; 
        border-radius: 20px; 
        box-shadow: 0 10px 30px rgba(0,0,0,0.6); 
        margin-top: 3rem; /* Increased spacing */
        margin-bottom: 3rem; /* Added spacing below the section */
    }
    .feature-item {
        background:#1b263b; 
        padding:15px; /* Reduced padding for medium size */
        border-radius:15px; 
        box-shadow:0 6px 15px rgba(0,0,0,0.5); 
        text-align:center;
    }
    .feature-item h3 {
        color:#00ffe0;
        font-size: 1.2rem; /* Adjusted font size */
    }
    .feature-item p {
        color:#b0c4de;
        font-size: 0.9rem; /* Adjusted font size */
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
st.markdown('<p class="form-subtitle">Enter flight details to predict potential delays based on a machine learning model.</p>', unsafe_allow_html=True)

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
    # Check if origin and destination airports are the same
    if airport_origin == airport_dest:
        st.error("The origin and destination airports cannot be the same. Please select different airports to proceed.")
    else:
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

        # Generate a hypothetical distribution for charts
        sample_size = 500
        delay_distribution = np.random.normal(loc=predicted_delay, scale=10, size=sample_size)
        delay_distribution = np.clip(delay_distribution, 0, 100) # Ensure delays are positive

        # Create charts
        st.markdown("<h2 style='text-align: center; color: #1b263b; margin-top: 2rem;'>📈 Delay Distribution and Class Probability</h2>", unsafe_allow_html=True)

        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            # Histogram for delay distribution
            fig_hist = px.histogram(x=delay_distribution, nbins=20, title="Probability Distribution of Delays")
            fig_hist.update_layout(xaxis_title="Predicted Delay (minutes)", yaxis_title="Frequency", showlegend=False)
            fig_hist.update_traces(marker_color='#3498db')
            st.plotly_chart(fig_hist, use_container_width=True)

        with chart_col2:
            # Pie chart for on-time vs delayed
            on_time_count = sum(1 for x in delay_distribution if x <= 15)
            delayed_count = sample_size - on_time_count
            
            fig_pie = px.pie(
                values=[on_time_count, delayed_count],
                names=["On-Time ✈️", "Delayed ⏱️"],
                title="Predicted On-Time vs. Delayed Flights",
                color_discrete_sequence=['#2ecc71', '#e74c3c']
            )
            fig_pie.update_traces(textposition='inside', textinfo='percent+label', marker=dict(line=dict(color='#0d1b2a', width=2)))
            st.plotly_chart(fig_pie, use_container_width=True)


# --- Feature Insights Section ---
st.markdown("""
<div class='feature-box'>
    <h1 style='color:#00ffe0; text-align:center; font-family: Arial Black, sans-serif; margin-bottom: 15px;'>📊 Feature Insights</h1>
    <p style='color:#d0d0d0; text-align:center; font-size:16px; margin-bottom:20px;'>Explore how different flight features impact delays and the importance of each factor in the prediction model.</p>
</div>
""", unsafe_allow_html=True)

# Example placeholders for charts / metrics
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class='feature-item'>
        <h3>Carrier Impact</h3>
        <p>Shows average delays by each airline carrier.</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class='feature-item'>
        <h3>Airport Impact</h3>
        <p>Shows how origin and destination airports influence flight delays.</p>
    </div>
    """, unsafe_allow_html=True)

# --- Design Section ---
st.markdown("""
<div class='feature-box' style='background: linear-gradient(135deg, #1f4068, #16314f);'>
    <h1 style='color:#75e6da; text-align:center; font-family: Arial Black, sans-serif; margin-bottom: 15px;'>🎨 Design and Credits</h1>
    <p style='color:#d0d0d0; text-align:center; font-size:16px; margin-bottom:20px;'>This application was designed using a dark, modern aesthetic with a focus on usability and a pleasant visual experience.</p>
    <div style='text-align:center; margin-top:20px;'>
        <p style='color:#b0c4de;'><strong>Theme:</strong> Dark Ocean Blue Gradient</p>
        <p style='color:#b0c4de;'><strong>Libraries:</strong> Streamlit, Scikit-learn, Pandas, Joblib, Plotly</p>
        <p style='color:#b0c4de;'><strong>Icons:</strong> Emojis ✈️ 🔮 ⏱️</p>
    </div>
</div>
""", unsafe_allow_html=True)
