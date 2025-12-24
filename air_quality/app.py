import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# ===============================
# STREAMLIT CONFIG
# ===============================
st.set_page_config(page_title="Air Quality Prediction", layout="wide")
st.title("🌍 Air Quality Prediction System")
st.write("Predict AQI using Indian city air-quality data")

# ===============================
# FILE UPLOAD (SAFE METHOD)
# ===============================
uploaded_file = st.file_uploader(
    "Upload city_day.csv file",
    type=["csv"]
)

if uploaded_file is None:
    st.warning("Please upload city_day.csv to continue")
    st.stop()

# ===============================
# LOAD DATA (MEMORY ONLY)
# ===============================
df = pd.read_csv(uploaded_file)

st.success("Dataset loaded successfully")

# ===============================
# CLEAN DATA
# ===============================
df.columns = df.columns.str.lower()

features = ["pm2.5", "pm10", "no2", "so2", "co", "o3"]
target = "aqi"

df = df[features + [target]]
df = df.dropna()

st.write(f"Total rows used: {df.shape[0]}")

# ===============================
# TRAIN / TEST SPLIT
# ===============================
X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ===============================
# MODEL
# ===============================
model = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# ===============================
# EVALUATION
# ===============================
y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

st.subheader("📊 Model Performance")
st.write(f"R² Score: {r2:.3f}")
st.write(f"Mean Absolute Error: {mae:.2f}")

# ===============================
# USER INPUT
# ===============================
st.subheader("🧪 Predict AQI")

col1, col2, col3 = st.columns(3)

with col1:
    pm25 = st.number_input("PM2.5", 0.0, 500.0, 50.0)
with col2:
    pm10 = st.number_input("PM10", 0.0, 500.0, 80.0)
with col3:
    no2 = st.number_input("NO2", 0.0, 300.0, 30.0)

col4, col5, col6 = st.columns(3)

with col4:
    so2 = st.number_input("SO2", 0.0, 200.0, 10.0)
with col5:
    co = st.number_input("CO", 0.0, 10.0, 1.0)
with col6:
    o3 = st.number_input("O3", 0.0, 300.0, 20.0)

if st.button("Predict AQI"):
    input_data = np.array([[pm25, pm10, no2, so2, co, o3]])
    prediction = model.predict(input_data)[0]

    st.subheader("✅ Prediction Result")

    if prediction <= 50:
        st.success(f"AQI: {prediction:.1f} (Good)")
    elif prediction <= 100:
        st.info(f"AQI: {prediction:.1f} (Satisfactory)")
    elif prediction <= 200:
        st.warning(f"AQI: {prediction:.1f} (Moderate)")
    else:
        st.error(f"AQI: {prediction:.1f} (Poor / Hazardous)")
