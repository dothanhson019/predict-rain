
import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="🌧️ Predict Rain App", layout="centered")
st.title("🌧️ Predict Rain (RainTomorrow)")

# Load models and encoders
scaler = joblib.load("saved_models/scaler.joblib")
pca = joblib.load("saved_models/pca_transformer.joblib")
rf_model = joblib.load("saved_models/random_forest_classifier_pca.joblib")
dt_model = joblib.load("saved_models/decision_tree_classifier_pca.joblib")
label_encoders = joblib.load("saved_models/label_encoders.joblib")
rain_encoder = label_encoders.get("RainTomorrow", None)

# Tạo form nhập dữ liệu
with st.form("input_form"):
    st.subheader("🔢 Input weather forecast data:")
    location = st.text_input("Location", "Sydney")
    min_temp = st.number_input("MinTemp (°C)", value=15.1)
    max_temp = st.number_input("MaxTemp (°C)", value=23.0)
    rainfall = st.number_input("Rainfall (mm)", value=0.4)
    evaporation = st.number_input("Evaporation (mm)", value=3.2)
    sunshine = st.number_input("Sunshine (hours)", value=9.8)
    wind_gust_dir = st.text_input("WindGustDir", "NW")
    wind_gust_speed = st.number_input("WindGustSpeed (km/h)", value=70.0)
    wind_dir_9am = st.text_input("WindDir9am", "WNW")
    wind_dir_3pm = st.text_input("WindDir3pm", "WNW")
    wind_speed_9am = st.number_input("WindSpeed9am", value=15.0)
    wind_speed_3pm = st.number_input("WindSpeed3pm", value=22.0)
    humidity_9am = st.number_input("Humidity9am (%)", value=67.0)
    humidity_3pm = st.number_input("Humidity3pm (%)", value=42.0)
    pressure_9am = st.number_input("Pressure9am (hPa)", value=1007.9)
    pressure_3pm = st.number_input("Pressure3pm (hPa)", value=1007.2)
    cloud_9am = st.slider("Cloud9am (0-8)", 0, 8, 6)
    cloud_3pm = st.slider("Cloud3pm (0-8)", 0, 8, 2)
    temp_9am = st.number_input("Temp9am (°C)", value=20.0)
    temp_3pm = st.number_input("Temp3pm (°C)", value=21.2)
    rain_today = st.selectbox("RainToday", ["No", "Yes"])
    model_type = st.selectbox("🧠Select a model", ["Random Forest", "Decision Tree"])
    submit = st.form_submit_button("Predict")
if submit:
    # Tạo DataFrame đầy đủ từ form nhập liệu
    full_input = pd.DataFrame([{
        "Location": location,
        "MinTemp": min_temp,
        "MaxTemp": max_temp,
        "Rainfall": rainfall,
        "Evaporation": evaporation,
        "Sunshine": sunshine,
        "WindGustDir": wind_gust_dir,
        "WindGustSpeed": wind_gust_speed,
        "WindDir9am": wind_dir_9am,
        "WindDir3pm": wind_dir_3pm,
        "WindSpeed9am": wind_speed_9am,
        "WindSpeed3pm": wind_speed_3pm,
        "Humidity9am": humidity_9am,
        "Humidity3pm": humidity_3pm,
        "Pressure9am": pressure_9am,
        "Pressure3pm": pressure_3pm,
        "Cloud9am": cloud_9am,
        "Cloud3pm": cloud_3pm,
        "Temp9am": temp_9am,
        "Temp3pm": temp_3pm,
        "RainToday": rain_today
    }])

    # Chọn đúng 10 feature có tương quan dương, đúng thứ tự huấn luyện
    selected_features = [
    "Humidity3pm", "RainToday", "Cloud3pm", "Humidity9am", "Cloud9am",
    "Rainfall", "WindGustSpeed", "WindSpeed9am", "WindSpeed3pm", "MinTemp"
    ]
    input_df = full_input[selected_features].copy()

    if 'RainToday' in input_df.columns:
        input_df['RainToday'] = input_df['RainToday'].map({'Yes': 1, 'No': 0})

    for col in input_df.columns:
        if col in label_encoders:
            input_df[col] = label_encoders[col].transform(input_df[col].astype(str))
    
    # Scale and PCA
    X_scaled = scaler.transform(input_df)
    X_pca = pca.transform(X_scaled)

    model = rf_model if model_type == "Random Forest" else dt_model
    prediction = model.predict(X_pca)[0]
    result_label = {0: "No", 1: "Yes"}.get(prediction, str(prediction))
    
    emoji = "☔" if prediction == 1 else "🌤️"
    st.success(f"🎯 Weather prediction result: **{emoji} {result_label}** (by {model_type})")