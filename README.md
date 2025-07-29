# 🏡 House Price Prediction Web App using Flask

This is a full-stack web application built using **Flask** that predicts house prices based on user-input features. The backend uses a trained machine learning model (likely Linear Regression or Random Forest), and the frontend is built using HTML templates.

## 📌 Project Overview

The goal of this project is to create a simple and interactive interface for predicting house prices based on various input features like area, number of bedrooms, bathrooms, and location.

### 🔧 Stack Used

- **Frontend**: HTML, CSS (via Flask `templates/`)
- **Backend**: Flask (Python micro web framework)
- **Machine Learning**: Scikit-learn
- **Model Deployment**: `joblib` or `pickle`

## 🛠️ Features

- User-friendly interface for inputting property details
- Predicts house prices in real-time
- Trained ML model integrated with Flask
- Lightweight and responsive

## 📁 File Structure

House-Price-Prediction-Using-Flask/
├── static/ # CSS or image files
├── templates/ # HTML templates (home.html, result.html)
├── model.pkl # Pre-trained ML model
├── app.py # Flask application
├── requirements.txt # Python dependencies
└── README.md # Project documentation
