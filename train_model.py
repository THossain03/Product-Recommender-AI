from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pickle
from data_processing import load_and_preprocess_data, get_features_targets

def train_model():
    # Load and preprocess the data
    X, y = load_and_preprocess_data()

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    # Initialize and train the model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Save the model and the preprocessing pipeline to a file
    with open('model.pkl', 'wb') as file:
        pickle.dump(model, file)

    print("Model trained and saved successfully!")

if __name__ == '__main__':
    train_model()
