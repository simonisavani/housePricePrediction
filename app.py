from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load('house_price_model.pkl')
# Load the scaler used during training
scaler = joblib.load('scaler.pkl')
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input features from the form
    locaton = request.form['location']
    bedrooms = float(request.form['bedrooms'])
    baths = float(request.form['baths'])
    balcony = float(request.form['balcony'])
    total_area = float(request.form['total_area'])
    #  // price/area
    price_per_sqft = float(request.form['price_per_sqft'])
    # // price_per_sqft/bedrooms
    price_per_bedroom = float(request.form['price_per_bedroom']) 
    area_per_bedroom = float(request.form['area_per_bedroom']) 
    # formula: baths/bedrooms
    bathroom_to_bedroom_ratio = float(request.form['bathroom_to_bedroom_ratio']) 

    # Make a prediction
    input_features = np.array([[bedrooms, baths, balcony, total_area, price_per_sqft,
                                price_per_bedroom, area_per_bedroom, bathroom_to_bedroom_ratio]])
    scaled_features = scaler.transform(input_features)
    prediction = model.predict(scaled_features)

    return render_template('index.html', prediction=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)
