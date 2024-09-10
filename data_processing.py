import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

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

    # Additional Features
    full_data['Interaction'] = full_data['InteractionType'].astype('category').cat.codes
    full_data['Product_Category'] = full_data['Category'].astype('category').cat.codes

    # Define features and target
    features = ['Gender', 'Price', 'Hour', 'Interaction', 'Product_Category']
    X = full_data[features]
    y = full_data['Rating']

    return X, y

def get_features_targets(full_data):
    # Define preprocessing pipeline
    numeric_features = ['Price', 'Hour']
    categorical_features = ['Gender', 'Interaction', 'Product_Category']

    # Preprocessing for numeric data
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())])

    # Preprocessing for categorical data
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)])

    # Apply transformations
    X = preprocessor.fit_transform(full_data)

    # Target variable
    y = full_data['Rating']

    return X, y
