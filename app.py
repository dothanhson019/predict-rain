
import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="🌧️ Predict Rain App", layout="centered")
st.title("🌧️ Dự đoán trời mưa (RainTomorrow)")

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
    min_temp = st.number_input("MinTemp (°C)", value=10.0)
    max_temp = st.number_input("MaxTemp (°C)", value=25.0)
    rainfall = st.number_input("Rainfall (mm)", value=5.0)
    evaporation = st.number_input("Evaporation (mm)", value=7.0)
    sunshine = st.number_input("Sunshine (hours)", value=7.0)
    wind_gust_dir = st.text_input("WindGustDir", "W")
    wind_gust_speed = st.number_input("WindGustSpeed (km/h)", value=30.0)
    wind_dir_9am = st.text_input("WindDir9am", "W")
    wind_dir_3pm = st.text_input("WindDir3pm", "WNW")
    wind_speed_9am = st.number_input("WindSpeed9am", value=15.0)
    wind_speed_3pm = st.number_input("WindSpeed3pm", value=20.0)
    humidity_9am = st.number_input("Humidity9am (%)", value=65.0)
    humidity_3pm = st.number_input("Humidity3pm (%)", value=55.0)
    pressure_9am = st.number_input("Pressure9am (hPa)", value=1012.0)
    pressure_3pm = st.number_input("Pressure3pm (hPa)", value=1010.0)
    cloud_9am = st.slider("Cloud9am (0-8)", 0, 8, 3)
    cloud_3pm = st.slider("Cloud3pm (0-8)", 0, 8, 4)
    temp_9am = st.number_input("Temp9am (°C)", value=20.0)
    temp_3pm = st.number_input("Temp3pm (°C)", value=23.0)
    rain_today = st.selectbox("RainToday", ["No", "Yes"])
    model_type = st.selectbox("🧠Select a model", ["Random Forest", "Decision Tree"])
    submit = st.form_submit_button("Predict")

if submit:
    input_df = pd.DataFrame([[
        location, min_temp, max_temp, rainfall, evaporation, sunshine,
        wind_gust_dir, wind_gust_speed, wind_dir_9am, wind_dir_3pm,
        wind_speed_9am, wind_speed_3pm, humidity_9am, humidity_3pm,
        pressure_9am, pressure_3pm, cloud_9am, cloud_3pm, temp_9am, temp_3pm,
        rain_today
    ]], columns=[
        'Location', 'MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine',
        'WindGustDir', 'WindGustSpeed', 'WindDir9am', 'WindDir3pm',
        'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm',
        'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm',
        'Temp9am', 'Temp3pm', 'RainToday'
    ])

    for col in input_df.columns:
        if col in label_encoders:
            input_df[col] = label_encoders[col].transform(input_df[col].astype(str))

    X_scaled = scaler.transform(input_df)
    X_pca = pca.transform(X_scaled)

    model = rf_model if model_type == "Random Forest" else dt_model
    prediction = model.predict(X_pca)[0]
    result_label = rain_encoder.inverse_transform([prediction])[0] if rain_encoder else str(prediction)

    emoji = "☔" if prediction == 1 else "🌤️"
    st.success(f"🎯 Kết quả dự đoán: **{{emoji}} {{result_label}}** (bằng {{model_type}})")
