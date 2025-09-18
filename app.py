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
        background: linear-gradient(45deg, #1b263b, #0d1b2a);
        color: #e0eaff;
        padding: 30px;
        border-radius: 15px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
        border: 2px solid #3498db;
        margin-bottom: 2rem;
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

# --- Input + Prediction Section (Redesigned) ---
st.markdown("""
<div class="form-box" style="background: linear-gradient(135deg, #1f2c34, #3a4a58); padding:30px; border-radius:20px; box-shadow: 0px 8px 25px rgba(0,0,0,0.7);">
</div>
""", unsafe_allow_html=True)

st.markdown("""
<h1 class="form-title" style="font-size:28px; text-align:center; color:#00ffe0; margin-bottom:10px; font-family: 'Arial Black', sans-serif;">
Flight Delay Predictor
</h1>
""", unsafe_allow_html=True)

st.markdown("""
<p class="form-subtitle" style="text-align:center; font-size:16px; color:#d0d0d0; margin-bottom:25px;">
Enter flight details to predict potential delays using a powerful machine learning model.
</p>
""", unsafe_allow_html=True)



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


# --- Feature Insights & Visuals Section (Standalone, Ready to Append) ---
import plotly.express as px
import pandas as pd

# Ensure session_state for storing multiple inputs exists
if 'user_inputs' in st.session_state and not st.session_state['user_inputs'].empty:
    user_df = st.session_state['user_inputs']

    st.markdown("""
    <div style='background: linear-gradient(135deg, #1a2a3a, #263d50); padding: 30px; border-radius: 20px; box-shadow: 0 10px 30px rgba(0,0,0,0.6); margin-top: 40px;'>
      <h1 style='color:#00ffe0; text-align:center; font-family: Arial Black, sans-serif;'>📊 Feature Insights & Visuals</h1>
      <p style='color:#d0d0d0; text-align:center; font-size:16px; margin-bottom:30px;'>Statistics and visualizations generated dynamically from the inputs you have provided.</p>
    </div>
    """, unsafe_allow_html=True)

    # --- Statistical Description ---
    st.markdown("""
    <div style='background:#1b263b; padding:20px; border-radius:15px; box-shadow:0 6px 15px rgba(0,0,0,0.5); margin-top:20px;'>
        <h3 style='color:#00ffe0; text-align:center;'>Statistical Summary of Input Features</h3>
    </div>
    """, unsafe_allow_html=True)
    st.dataframe(user_df.describe().T.style.format('{:.2f}'))

    # --- Carrier Impact ---
    carrier_chart = user_df.groupby('carrier_name')['predicted_delay'].mean().reset_index()
    fig1 = px.bar(carrier_chart, x='carrier_name', y='predicted_delay', color='predicted_delay', color_continuous_scale='Viridis', title='Average Predicted Delay by Carrier')
    st.plotly_chart(fig1, use_container_width=True)

    # --- Origin Airport Impact ---
    origin_chart = user_df.groupby('airport_origin')['predicted_delay'].mean().reset_index()
    fig2 = px.bar(origin_chart, x='airport_origin', y='predicted_delay', color='predicted_delay', color_continuous_scale='Cividis', title='Average Predicted Delay by Origin Airport')
    st.plotly_chart(fig2, use_container_width=True)

    # --- Predicted Delay Distribution ---
    fig3 = px.histogram(user_df, x='predicted_delay', nbins=20, color='predicted_class', title='Predicted Delay Distribution', color_discrete_map={'On-Time ✈️':'#00ffcc','Delayed ⏱️':'#ffd700'})
    st.plotly_chart(fig3, use_container_width=True)
