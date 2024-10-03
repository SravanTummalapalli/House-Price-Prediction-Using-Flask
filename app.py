import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd

app = Flask(__name__)   
model = pickle.load(open('house_price_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    # Extracting the input features
    area = float(request.form.get('area', 0))
    bedrooms = float(request.form.get('bedrooms', 1))
    bathrooms = float(request.form.get('bathrooms', 2))
    stories = float(request.form.get('stories', 3))
    parking = float(request.form.get('parking', 4))

    mainroad_yes = 1 if request.form.get('mainroad_yes') == 'yes' else 0
    guestroom_yes = 1 if request.form.get('guestroom_yes') == 'yes' else 0
    basement_yes = 1 if request.form.get('basement_yes') == 'yes' else 0
    hotwaterheating_yes = 1 if request.form.get('hotwaterheating_yes') == 'yes' else 0
    airconditioning_yes = 1 if request.form.get('airconditioning_yes') == 'yes' else 0
    prefarea_yes = 1 if request.form.get('prefarea_yes') == 'yes' else 0

    # Furnishing status options
    furnishingstatus_semi_furnished = 1 if request.form.get('furnishingstatus') == 'semi-furnished' else 0
    furnishingstatus_unfurnished = 1 if request.form.get('furnishingstatus') == 'unfurnished' else 0

    # Correcting the furnishing status feature name
    furnishingstatus_semi_furnished = 1 if request.form.get('furnishingstatus') == 'semi-furnished' else 0
    furnishingstatus_unfurnished = 1 if request.form.get('furnishingstatus') == 'unfurnished' else 0

    # Creating a list of features in the correct order
    features = [
        area, bedrooms, bathrooms, stories, parking,
        mainroad_yes, guestroom_yes, basement_yes,
        hotwaterheating_yes, airconditioning_yes,
        prefarea_yes, furnishingstatus_semi_furnished, furnishingstatus_unfurnished
    ]

    # Use the exact names as used during model training
    feature_names = [
        'area', 'bedrooms', 'bathrooms', 'stories', 'parking',
        'mainroad_yes', 'guestroom_yes', 'basement_yes',
        'hotwaterheating_yes', 'airconditioning_yes',
        'prefarea_yes', 'furnishingstatus_semi-furnished', 'furnishingstatus_unfurnished'
    ]

    input_data = pd.DataFrame([features], columns=feature_names)
    print(f'Input DataFrame:\n{input_data}')

    # Predicting the house price
    prediction = model.predict(input_data)[0]
    
    return render_template('index.html', prediction_text=f'Predicted House Price: {prediction:.2f}')

if __name__ == "__main__":
    app.run(debug=True)
