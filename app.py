import numpy as np
import pandas as pd
import pickle
from flask import Flask, request, jsonify
from flask_cors import CORS  # Import the CORS library

# --- 1. Initialize Flask App ---
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# --- 2. Load the Trained Model ---
model_path = 'pipe.pkl'
try:
    pipe = pickle.load(open(model_path, 'rb'))
except FileNotFoundError:
    print(f"Error: Model file not found at '{model_path}'.")
    print("Please run the 'train_model.py' script first to generate the model file.")
    exit()


# --- 3. Define API Endpoints ---

@app.route('/')
def home():
    """A simple endpoint to check if the API is running."""
    return "<h1>IPL Win Predictor API</h1><p>The API is running. Use the /predict endpoint to get predictions.</p>"


@app.route('/predict', methods=['POST'])
def predict():
    """
    Receives match data in JSON format, uses the loaded model to predict
    the win probability, and returns the result.
    """
    try:
        data = request.get_json(force=True)

        # Create a DataFrame from the incoming JSON data in the correct order
        input_df = pd.DataFrame({
            'batting_team': [data['batting_team']],
            'bowling_team': [data['bowling_team']],
            'city': [data['city']],
            'runs_left': [data['runs_left']],
            'balls_left': [data['balls_left']],
            'wickets_left': [data['wickets_left']],
            'total_runs_x': [data['target']],
            'crr': [data['crr']],
            'rrr': [data['rrr']]
        })

        # Make prediction using the loaded pipeline
        result = pipe.predict_proba(input_df)

        # Format the response, converting numpy floats to standard Python floats
        response = {
            'bowling_team_win_prob': float(result[0][0]),
            'batting_team_win_prob': float(result[0][1])
        }

        return jsonify(response)

    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'error': 'An error occurred during prediction. Check the server logs.'}), 500


# --- 4. Run the App ---

if __name__ == '__main__':
    # To run this server:
    # 1. First, run 'train_model.py' to create 'pipe.pkl'.
    # 2. Make sure you have Flask and Flask-Cors installed: pip install Flask Flask-Cors
    # 3. Run this script: python app.py
    app.run(debug=True, port=5000)
