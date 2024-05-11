from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pickle
from data_processing import load_and_preprocess_data, get_features_targets

def train_model():
    full_data = load_and_preprocess_data()
    X, y = get_features_targets(full_data)
    
    # Assuming all necessary preprocessing steps are complete
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Save the model to a file
    with open('model.pkl', 'wb') as file:
        pickle.dump(model, file)

    print("Model trained and saved successfully!")

if __name__ == '__main__':
    train_model()
