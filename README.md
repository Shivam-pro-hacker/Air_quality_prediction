# 🌍 Air Quality Prediction System

> **An ML-powered Streamlit application that predicts Air Quality Index (AQI) from real Indian urban air-pollution data.**

This project builds an end-to-end machine learning pipeline that learns AQI behavior from historical pollutant measurements (PM2.5, PM10, NO₂, SO₂, CO, O₃) and provides interactive AQI prediction through a web interface.

---

## 📌 Overview

Air pollution in urban India poses serious environmental and health risks.  
This system applies machine learning to estimate the Air Quality Index (AQI) from pollutant concentrations, enabling quick assessment of air quality conditions.

The model is trained on real CPCB-based pollution data and deployed using Streamlit for interactive prediction.

---

## 🚀 Features

- Machine Learning–based AQI prediction  
- Random Forest regression model  
- Interactive Streamlit web interface  
- Real Indian pollution dataset training  
- Model performance metrics (R², MAE)  
- Full pollutant input prediction  
- Pollutant full-form explanations  
- Cross-platform (Windows & Linux)  

---

## 🧠 Machine Learning Model

**Algorithm:** Random Forest Regressor  

**Input Features:**

- PM2.5  
- PM10  
- NO₂  
- SO₂  
- CO  
- O₃  

**Target:** AQI  

**Typical Performance:**

- R² Score ≈ 0.90–0.93  
- MAE ≈ 15–18 AQI units  

The model captures nonlinear relationships between pollutants and AQI using ensemble decision trees.

---

## 🌫 Pollutants Used

- **PM2.5** — Fine particulate matter ≤ 2.5 µm  
- **PM10** — Coarse particulate matter ≤ 10 µm  
- **NO₂** — Nitrogen Dioxide  
- **SO₂** — Sulfur Dioxide  
- **CO** — Carbon Monoxide  
- **O₃** — Ground-level Ozone  

These pollutants are primary contributors to AQI.

---

## 📊 AQI Categories (India CPCB)

| AQI Range | Category |
|-----------|---------|
| 0 – 50 | Good |
| 51 – 100 | Satisfactory |
| 101 – 200 | Moderate |
| 201 – 300 | Poor |
| 301 – 400 | Very Poor |
| 401 – 500 | Severe |

---

## ▶️ Installation

### 🪟 Windows

```bash
# 1. Install Python 3.9+
https://www.python.org/downloads/
# Enable Add Python to PATH during install

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the application
python -m streamlit run app.py


