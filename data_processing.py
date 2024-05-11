import pandas as pd

def load_and_preprocess_data():
    user_data = pd.read_csv('user_data.csv')
    product_data = pd.read_csv('product_data.csv')
    interaction_data = pd.read_csv('interaction_data.csv')

    # Data Preprocessing
    full_data = pd.merge(interaction_data, user_data, on='UserID')
    full_data = pd.merge(full_data, product_data, on='ProductID')
    full_data['Gender'] = full_data['Gender'].apply(lambda x: 1 if x == 'M' else 0)
    full_data['Price'] = (full_data['Price'] - full_data['Price'].min()) / (full_data['Price'].max() - full_data['Price'].min())
    full_data['Timestamp'] = pd.to_datetime(full_data['Timestamp'])
    full_data['Hour'] = full_data['Timestamp'].dt.hour

    return full_data

def get_features_targets(full_data):
    # Assuming the data includes a 'Rating' column for interactions
    X = full_data.drop(['Rating'], axis=1)
    y = full_data['Rating']
    return X, y
