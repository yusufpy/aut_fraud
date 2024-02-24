from flask import Flask, request, jsonify
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the trained model
with open("orest.pkl", "rb") as f:
    model = pickle.load(f)

# Initialize Flask app
app = Flask(__name__)

# Define a route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get the JSON data from the request
    data = request.get_json()
    
    # Load the data into a DataFrame
    df = pd.DataFrame(data, index=[0])
    # Standardize the numerical columns
    scaler = StandardScaler()
    numerical_cols = ['amount', 'oldbalanceOrg', 'newbalanceOrig']
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    
    # Make prediction
    prediction = model.predict(df)
    
    # Return the prediction as JSON
    return jsonify({'prediction': int(prediction[0])})

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
