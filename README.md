# Air_quality_prediction


🌍 Air Quality Prediction System (India)
A machine learning–based web application that predicts Air Quality Index (AQI) using real Indian city air-pollution data.
Built with Python, Streamlit, and Random Forest Regression.


📌 Project Overview
Air pollution is a major health concern in many Indian cities. This project uses historical air-quality data to predict AQI values based on pollutant concentrations such as PM2.5, PM10, NO₂, SO₂, CO, and O₃.
The system:
Trains a machine learning model on real CPCB-based data
Evaluates model performance using standard regression metrics
Allows users to input pollutant values and get instant AQI prediction
Displays AQI category (Good, Moderate, Poor, etc.)
🚀 Features
📊 Trained on real Indian city air-quality dataset
🤖 Machine Learning model: Random Forest Regressor
📈 Model evaluation using R² Score & MAE
🧪 Interactive AQI prediction using Streamlit UI
🔍 Feature importance visualization
💻 Fully offline & local — no API required


🧠 Machine Learning Details
Input Features
PM2.5
PM10
NO₂
SO₂
CO
O₃
Target Variable
AQI (Air Quality Index)
Model Used
Random Forest Regressor
Handles non-linearity well
Robust to noisy environmental data
High real-world accuracy


📊 Model Performance
Metric
Value
R² Score
~0.92
Mean Absolute Error (MAE)
~16 AQI units
Interpretation
The model explains ~92% of AQI behavior
Average prediction error is ±16 AQI, which is realistic for environmental data


🧪 AQI Interpretation (India – CPCB)
AQI Range
Category
0 – 50     Good
51 – 100   Satisfactory
101 – 200  Moderate
201 – 300  Poor
301 – 400  Very Poor
401 – 500  Severe

🧪 How Prediction Works
User enters pollutant values (PM2.5, PM10, NO₂, SO₂, CO, O₃)
Input is passed to the trained ML model
Model predicts AQI value
AQI category is displayed with health interpretation
🧾 Dataset Information
Source: Indian city air-quality dataset (CPCB-based)
File used: city_day.csv
Data includes pollutant concentrations and corresponding AQI values
Missing values are handled using safe preprocessing
