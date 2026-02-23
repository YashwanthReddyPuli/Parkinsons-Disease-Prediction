from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

app = Flask(__name__)

CORS(app) 


try:
    model = joblib.load('pd_bagging_model.pkl')
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        avg_step = float(data.get('avg_step_length', 0))
        std_step = float(data.get('std_step_length', 0))
        avg_arm = float(data.get('avg_arm_swing', 0))
        
        stability = avg_step / (std_step + 1e-6)
        features = np.array([[avg_step, std_step, avg_arm, stability]])
        
        
        if stability > 3.0 and std_step < 8 and avg_arm > 55:
            final_idx = 0 
            
        
        elif stability < 1.0 or std_step > 25:
            final_idx = 3 # Force Severe
            
        
        elif stability < 1.4 or avg_arm < 40:
            final_idx = 2 
            
        
        elif stability < 2.0:
            final_idx = 1 #
        else:
            final_idx = int(model.predict(features)[0])

        severity_labels = {0: 'Normal', 1: 'Mild', 2: 'Moderate', 3: 'Severe'}
        result = severity_labels.get(final_idx, "Unknown")

        return jsonify({'prediction': result, 'stability_index': round(stability, 2), 'status': 'Success'})
        
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'Failed'}), 500

if __name__ == '__main__':
    app.run(port=5000, debug=True)