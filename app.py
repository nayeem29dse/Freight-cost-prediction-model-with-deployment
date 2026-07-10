import pickle
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

with open('freight_model.pkl', 'rb') as f:
    reg_model = pickle.load(f)

with open('freight_classifier.pkl', 'rb') as f:
    clf_model = pickle.load(f)

with open('label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)

@app.route('/')
def home():
    return jsonify({
        "message": "Freight Cost Prediction API",
        "endpoints": {
            "/predict": "Predict freight cost in USD",
            "/classify": "Flag high freight risk orders"
        }
    })

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    try:
        vendor_encoded = le.transform([data['VendorName']])[0]
        features = np.array([[data['Quantity'], data['Dollars'], data['po_month'], vendor_encoded]])
        prediction = reg_model.predict(features)[0]
        return jsonify({"predicted_freight": round(float(prediction), 2), "currency": "USD"})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/classify', methods=['POST'])
def classify():
    data = request.get_json()
    try:
        vendor_encoded = le.transform([data['VendorName']])[0]
        features = np.array([[data['Quantity'], data['Dollars'], vendor_encoded]])
        proba = clf_model.predict_proba(features)[0][1]
        return jsonify({"high_freight_risk": bool(proba >= 0.1), "probability": round(float(proba), 4)})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)