from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import pickle
import eventlet
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

app = Flask(__name__, template_folder='templates')

# Function to load data on boot
def load_data_to_app():
    # Load the model and preprocessing pipeline
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)
    
    # Load the preprocessing pipeline
    with open('preprocessor.pkl', 'rb') as preprocessor_file:
        preprocessor = pickle.load(preprocessor_file)

    user_data = pd.read_csv('user_data.csv')
    product_data = pd.read_csv('product_data.csv')

    return model, preprocessor, user_data, product_data

@app.before_first_request
def initialize_data():
    print("Loading data before first request...")
    app.model, app.preprocessor, app.user_data, app.product_data = load_data_to_app()

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    user_id = int(request.form['user_id'])
    top_n = int(request.form.get('top_n', 5))
    location = request.form['location']

    user_info = app.user_data[app.user_data['UserID'] == user_id]

    if user_info.empty:
        return jsonify({"error": "User not found"})

    # Generate user-product features for prediction
    user_features = user_info.drop(['UserID'], axis=1)
    products_features = app.product_data.drop(['ProductID'], axis=1)
    
    location_filtered_products = app.product_data[app.product_data['Location'] == location]
    if location_filtered_products.empty:
        return jsonify({"error": "No products found for this location"})

    user_product_features = user_features.assign(key=1).merge(location_filtered_products.assign(key=1), on='key').drop('key', 1)

    # Apply preprocessing
    X_processed = app.preprocessor.transform(user_product_features)

    # Predict ratings or preferences
    predictions = app.model.predict(X_processed)
    top_product_indices = np.argsort(-predictions)[:top_n]
    top_products = location_filtered_products.iloc[top_product_indices]

    recommendations = [
        {"name": row['Name'], "price": row['Price']} for index, row in top_products.iterrows()
    ]

    return jsonify({"recommendations": recommendations})

@app.route('/validate_user/<int:user_id>', methods=['GET'])
def validate_user(user_id):
    user_exists = not app.user_data[app.user_data['UserID'] == user_id].empty
    return jsonify({"exists": user_exists})

if __name__ == '__main__':
    app.run()
    eventlet.wsgi.server(eventlet.listen(('192.168.2.11', 8000)), app)
