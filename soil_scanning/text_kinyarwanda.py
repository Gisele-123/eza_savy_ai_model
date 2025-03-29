import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from joblib import load
import requests
import colorsys

try:
    model = load('plant_model.pkl')
    scaler = load('soil_encoder.pkl')
    le_plant = load('plant_encoder.pkl')
except FileNotFoundError as e:
    print(f"Error loading model files: {e}")
    exit()

class BilingualSystem:
    def __init__(self):
        self.translator = KinyarwandaTranslator("c6bd406b1cmshfc8def9c506f3fep1e844fjsn98a2c8a4099c")
        self.color_options = {
            "red": {"rgb": (255, 0, 0), "rw": "umutuku"},
            "black": {"rgb": (0, 0, 0), "rw": "umukara"},
            "alluvial": {"rgb": (210, 180, 140), "rw": "ikijuju"},
            "clay": {"rgb": (189, 161, 137), "rw": "ikigina"}
        }

    def display_message(self, english_text):
        """Show messages in both English and Kinyarwanda"""
        kinyarwanda_text = self.translator.translate_to_kinyarwanda(english_text)
        print(f"\n[EN] {english_text}")
        print(f"[RW] {kinyarwanda_text}\n")

    def get_soil_color(self):
        """Get soil color with flexible input options"""
        while True:
            self.display_message("Please choose your soil color:")
            print("Available colors / Amabara y'ubutaka:")
            
            # Display color options
            for i, (color_name, data) in enumerate(self.color_options.items(), 1):
                print(f"{i}. {color_name.ljust(8)} ({data['rw']})")
            
            self.display_message("Enter the color name or number:")
            user_input = input("> ").lower().strip()
            
            # Check for number input
            if user_input.isdigit():
                num = int(user_input)
                if 1 <= num <= len(self.color_options):
                    selected = list(self.color_options.values())[num-1]
                    color_name = list(self.color_options.keys())[num-1]
                    self.display_message(f"Selected: {color_name} soil")
                    return selected["rgb"]
            
            # Check for text input
            for color_name, data in self.color_options.items():
                if user_input in [color_name, data['rw']]:
                    self.display_message(f"Selected: {color_name} soil")
                    return data["rgb"]
            
            self.display_message("Invalid input. Please try again.")

    def predict_and_display(self, R, G, B, top_n=3):
        """Make prediction and show results in both languages"""
        # Prepare input features
        h, l, s = colorsys.rgb_to_hls(R/255, G/255, B/255)
        input_data = np.array([[R, G, B, h, l, s]])
        scaled_features = scaler.transform(input_data)
        
        # Get predictions
        probabilities = model.predict_proba(scaled_features)[0]
        top_indices = np.argsort(probabilities)[-top_n:][::-1]
        
        # Convert numeric predictions to plant names
        plant_names = le_plant.inverse_transform(top_indices)
        
        # Prepare results
        results = {
            "plants": plant_names,
            "confidences": [probabilities[i] * 100 for i in top_indices]
        }
        
        # Display results
        self.display_message("=== Recommended Crops ===")
        for i, (plant, confidence) in enumerate(zip(results['plants'], results['confidences']), 1):
            plant_rw = self.translator.translate_to_kinyarwanda(plant)
            print(f"{i}. [EN] {str(plant).ljust(15)} ({confidence:.1f}% suitable)")
            print(f"   [RW] {str(plant_rw).ljust(15)} (amahirwe {confidence:.1f}%)")
        
        best_plant = results['plants'][0]
        best_plant_rw = self.translator.translate_to_kinyarwanda(best_plant)
        self.display_message(f"Best crop: {best_plant}")
        self.display_message(f"Igihingwa cy'ibanze: {best_plant_rw}")

# ================== TRANSLATION SERVICE ==================
class KinyarwandaTranslator:
    def __init__(self, api_key):
        self.api_key = api_key
        self.url = "https://du-mt-api.p.rapidapi.com/translate"
        self.headers = {
            "X-RapidAPI-Key": api_key,
            "X-RapidAPI-Host": "du-mt-api.p.rapidapi.com"
        }

    def translate_to_kinyarwanda(self, text):
        try:
            payload = {"text": str(text), "source": "en", "target": "rw"}  # Ensure text is string
            response = requests.post(self.url, json=payload, headers=self.headers)
            return response.json().get('translatedText', str(text))
        except Exception as e:
            print(f"Translation error: {e}")
            return str(text)  # Fallback to string representation

# ================== MAIN APPLICATION ==================
def main():
    system = BilingualSystem()
    
    print("\n" + "="*50)
    system.display_message("=== EzaSavvy ===")
    system.display_message("=== Sisitemu ya EzaSavvy ===")
    print("="*50)
    
    # Get soil color
    R, G, B = system.get_soil_color()
    
    # Get and display predictions
    system.display_message("Analyzing your soil...")
    system.display_message("Ndareba ibara ry'ubutaka bwawe...")
    system.predict_and_display(R, G, B)
    
    system.display_message("Thank you for using our system!")
    system.display_message("Murakoze gukoresha sisitemu yacu!")

if __name__ == "__main__":
    main()