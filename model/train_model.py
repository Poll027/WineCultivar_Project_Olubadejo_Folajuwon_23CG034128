import pandas as pd
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib
import os

def train_and_save_model():
    print("Loading Wine dataset...")
    wine = load_wine()
    data = pd.DataFrame(data=wine.data, columns=wine.feature_names)
    data['target'] = wine.target

    # Select specific features as per notebook analysis
    selected_features = [
        "alcohol",
        "malic_acid",
        "alcalinity_of_ash",
        "total_phenols",
        "color_intensity",
        "proline"
    ]
    
    print(f"Selected features: {selected_features}")
    X = data[selected_features]
    y = data['target']

    # Initialize Scaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train/Test Split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train SVC Model
    print("Training SVC model...")
    model = SVC(kernel='rbf', probability=True) # probability=True good for confidence scores if needed later
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {acc:.4f}")

    # Ensure model directory exists
    os.makedirs('model', exist_ok=True)

    # Save Model and Scaler at the root as requested by user structure, or in model/ folder?
    # User asked for |- /model/ |- wine_cultivar_model.pkl
    # So I will save to model/ directory.
    
    model_path = os.path.join('model', 'wine_cultivar_model.pkl')
    scaler_path = os.path.join('model', 'wine_scaler.pkl')

    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)

    print(f"Model saved to {model_path}")
    print(f"Scaler saved to {scaler_path}")

if __name__ == "__main__":
    train_and_save_model()
