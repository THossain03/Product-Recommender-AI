from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import pickle
import eventlet

app = Flask(__name__, template_folder='templates')

# Function to load data on boot
def load_data_to_app():
    # Load the model and data
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)
    user_data = pd.read_csv('user_data.csv')
    product_data = pd.read_csv('product_data.csv')
    print("Begin loading data")
    return model, user_data, product_data

@app.before_first_request
def initialize_data():
    print("Loading data before first request...")
    app.model, app.user_data, app.product_data = load_data_to_app()

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    user_id = int(request.form['user_id'])
    top_n = int(request.form.get('top_n', 5))  # Default to 5 if not provided
    user_info = app.user_data[app.user_data['UserID'] == user_id]

    if user_info.empty:
        return render_template('index.html', recommendation=None, error="User not found")

    # Generate user-product features for prediction
    user_features = user_info.drop(['UserID'], axis=1)
    products_features = app.product_data.drop(['ProductID'], axis=1)
    user_product_features = user_features.assign(key=1).merge(products_features.assign(key=1), on='key').drop('key', 1)

    # Predict ratings or preferences taking top n recommendations
    predictions = app.model.predict(user_product_features)
    top_product_indices = np.argsort(-predictions)[:top_n]
    top_products = app.product_data.iloc[top_product_indices]

    recommendations = [
        {"name": row['Name'], "price": row['Price']} for index, row in top_products.iterrows()
    ]

    return render_template('index.html', recommendation=recommendations)

if __name__ == '__main__':
    app.run()
    eventlet.wsgi.server(eventlet.listen(('192.168.2.11', 8000)), app)
