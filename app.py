from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import joblib
import json
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)

# Load models and encoders
plant_model = joblib.load('soil_scanning/plant_model.pkl')
le_plant = joblib.load('soil_scanning/plant_encoder.pkl')
le_soil = joblib.load('soil_scanning/soil_encoder.pkl')

# Initialize scaler (you should save this during training)
scaler = MinMaxScaler()
env_features = ['pH', 'Temperature', 'Rainfall', 'Light_Hours']
# Note: You should fit this with your training data and save it
# For now we'll initialize with dummy values
scaler.fit(np.array([[6.5, 25, 150, 12]]))  # Replace with your actual training data

def load_weather_forecast():
    try:
        with open('weather/weather_prediction/shared_weather_data.json') as f:
            data = json.load(f)
        
        forecast = data['current_forecast']
        return {
            'temperature': float(forecast['temperature']),
            'rainfall': float(forecast['rainfall']),
            'date': forecast.get('date', 'Day 5')
        }
    except Exception as e:
        print(f"Weather loading error: {str(e)}")
        return {
            'temperature': 25.0,
            'rainfall': 150.0,
            'date': 'Default'
        }

@app.route('/weather_forecast', methods=['GET'])
def get_forecast():
    try:
        with open('weather/weather_prediction/shared_weather_data.json') as f: 
            data = json.load(f)
        return jsonify({
            'success': True,
            'current': data['current_forecast'],
            'forecast': data['full_forecast']
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        # Get weather data
        weather = load_weather_forecast()
        
        # Extract parameters with defaults
        R = int(data.get('R', 120))
        G = int(data.get('G', 80))
        B = int(data.get('B', 50))
        pH = float(data.get('pH', 6.5))
        light_hours = float(data.get('light_hours', 10))
        temperature = weather['temperature']
        rainfall = weather['rainfall']
        
        # Calculate derived features
        hue = np.arctan2(np.sqrt(3) * (G - B), 2*R - G - B)
        saturation = 1 - (3 * min(R,G,B)) / (R + G + B + 1e-6)
        brightness = (R + G + B) / 3
        
        # Prepare input dataframe
        input_df = pd.DataFrame([[
            R, G, B, hue, saturation, brightness, 
            pH, temperature, rainfall, light_hours
        ]], columns=[
            'R', 'G', 'B', 'Hue', 'Saturation', 'Brightness',
            'pH', 'Temperature', 'Rainfall', 'Light_Hours'
        ])
        
        # Scale environmental features
        input_df[env_features] = scaler.transform(input_df[env_features])
        
        # Get predictions
        probas = plant_model.predict_proba(input_df)[0]
        top_n = 5
        top_idx = np.argsort(probas)[-top_n:][::-1]
        percentages = (probas[top_idx] * 100).round(1)
        plant_names = le_plant.inverse_transform(top_idx)
        
        # Prepare response
        recommendations = [
            {
                "plant": plant_names[i],
                "confidence": float(percentages[i]),
                "suitability": (
                    "Excellent" if p > 80 else
                    "Good" if p > 60 else
                    "Moderate" if p > 40 else
                    "Marginal"
                )
            } for i, p in enumerate(percentages)
        ]
        
        return jsonify({
            "success": True,
            "recommendations": recommendations,
            "weather": {
                "temperature": temperature,
                "rainfall": rainfall,
                "date": weather['date']
            },
            "soil": {
                "color": [R, G, B],
                "pH": pH,
                "light_hours": light_hours
            }
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)