
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

# Tạo form nhập liệu
with st.form("input_form"):
    st.subheader("🔢 Nhập dữ liệu thời tiết:")
    min_temp = st.number_input("MinTemp (°C)", value=10.0)
    max_temp = st.number_input("MaxTemp (°C)", value=25.0)
    rainfall = st.number_input("Rainfall (mm)", value=5.0)
    evaporation = st.number_input("Evaporation (mm)", value=7.0)
    sunshine = st.number_input("Sunshine (hours)", value=7.0)
    wind_gust_speed = st.number_input("WindGustSpeed (km/h)", value=30.0)
    humidity_3pm = st.number_input("Humidity3pm (%)", value=60.0)
    pressure_9am = st.number_input("Pressure9am (hPa)", value=1012.0)
    temp_3pm = st.number_input("Temp3pm (°C)", value=23.0)
    rain_today = st.selectbox("RainToday", ["No", "Yes"])
    submit = st.form_submit_button("Dự đoán")

if submit:
    input_data = pd.DataFrame([[
        min_temp, max_temp, rainfall, evaporation, sunshine,
        wind_gust_speed, humidity_3pm, pressure_9am, temp_3pm,
        1 if rain_today == "Yes" else 0
    ]], columns=[
        "MinTemp", "MaxTemp", "Rainfall", "Evaporation", "Sunshine",
        "WindGustSpeed", "Humidity3pm", "Pressure9am", "Temp3pm",
        "RainToday"
    ])

    # Chuẩn hóa và PCA
    X_scaled = scaler.transform(input_data)
    X_pca = pca.transform(X_scaled)

    model_type = st.selectbox("🧠 Chọn mô hình", ["Random Forest", "Decision Tree"])
    model = rf_model if model_type == "Random Forest" else dt_model

    prediction = model.predict(X_pca)[0]
    result_label = rain_encoder.inverse_transform([prediction])[0] if rain_encoder else str(prediction)

    emoji = "☔" if prediction == 1 else "🌤️"
    st.success(f"🎯 Kết quả dự đoán: **{emoji} {result_label}** (bằng {model_type})")
