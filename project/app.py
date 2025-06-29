from flask import Flask, render_template, request
import pickle
import numpy as np
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)

app = Flask(_name_)

# Paths to model and scaler files
base_dir = os.path.dirname(os.path.abspath(_file_))
model_path = os.path.join(base_dir, 'rf_acc_68.pkl')
scaler_path = os.path.join(base_dir, 'normalizer.pkl')

# Load model and scaler
try:
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
except FileNotFoundError as e:
    logging.error(f"❌ File loading error: {e}")
    model = None
    scaler = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or scaler is None:
        return render_template('result.html', prediction_text="❌ Model or Scaler file not found.")
    try:
        gender_input = request.form['gender'].strip().lower()
        if gender_input not in ['male', 'female']:
            return render_template('result.html', prediction_text="❌ Please enter gender as 'male' or 'female'.")
        gender = 1 if gender_input == 'male' else 0

        # Extract and format features
        features = [
            float(request.form['age']),
            gender,
            float(request.form['total_bilirubin']),
            float(request.form['direct_bilirubin']),
            float(request.form['alkaline_phosphotase']),
            float(request.form['alamine_aminotransferase']),
            float(request.form['aspartate_aminotransferase']),
            float(request.form['total_proteins']),
            float(request.form['albumin']),
            float(request.form['albumin_globulin_ratio'])
        ]

        input_data = np.array([features])
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]

        result = '🟥 Liver Disease Detected' if prediction == 1 else '🟩 No Liver Disease'
        return render_template('result.html', prediction_text=result)

    except Exception as e:
        logging.error(f"❌ Prediction error: {str(e)}")
        return render_template('result.html', prediction_text=f"❌ Error: {str(e)}")

if _name_ == '_main_':
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host='0.0.0.0', port=port)