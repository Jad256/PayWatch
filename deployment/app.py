from flask import Flask, request, jsonify
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load the trained model
model_path = r'C:\Users\Jad_S\Documents\Python Scripts\Credit card Fraud Model\models\Trained_model.pkl'
with open(model_path, 'rb') as file:
    model = pickle.load(file)

# Load the scaler
scaler_path = r'C:\Users\Jad_S\Documents\Python Scripts\Credit card Fraud Model\data\scaler.pkl'
with open(scaler_path, 'rb') as file:
    scaler = pickle.load(file)

# Define a route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get data from POST request
    data = request.get_json(force=True)
    # Convert data to numpy array
    data = np.array(list(data.values())).reshape(1, -1)
    # Scale the input data
    data_scaled = scaler.transform(data)
    # Make prediction
    prediction = model.predict(data_scaled)
    # Return prediction as JSON
    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)
