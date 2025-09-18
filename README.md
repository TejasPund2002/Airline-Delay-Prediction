# Airline Delay Prediction

## Overview
The **Airline Delay Prediction** project is a machine learning application that predicts flight arrival delays based on multiple factors such as weather, traffic, airport congestion, and flight schedules. The project provides both **regression (delay in minutes)** and **classification (delay class)** predictions through a user-friendly **Streamlit web application**.

This project is designed for airlines, passengers, and aviation enthusiasts to **analyze and anticipate flight delays**, enabling better planning and decision-making.

---

## Features

- **Interactive Web App:** Built with Streamlit for quick predictions and visualization.
- **Regression Prediction:** Predicts the estimated flight delay in minutes.
- **Delay Classification:** Categorizes delays into classes such as:
  - On-time
  - Short Delay
  - Long Delay
- **Dynamic Input Handling:** Users can enter flight details like carrier, origin/destination airport, weather, traffic level, and time of flight.
- **Data Exploration:** Includes basic EDA and visualization for insights into factors affecting delays.
- **Scalable Machine Learning Model:** Random Forest Regressor trained on real-world flight data.

---

## Dataset
- **Dataset:** `flight_delay_dataset.csv`
- **Source:** Collected from historical flight data (internal dataset for demonstration).
- **Features Include:**
  - `carrier_name`
  - `airport_origin`
  - `airport_dest`
  - `flight_date`
  - `hour`, `day_of_week`, `month`
  - `weather_condition`
  - `traffic_level`
  - `airport_congestion_index`
  - `arr_delay` (target variable)

---

## Installation

1. Clone the repository:
```bash
git clone https://github.com/TejasPund2002/Airline-Delay-Prediction.git
cd Airline-Delay-Prediction
