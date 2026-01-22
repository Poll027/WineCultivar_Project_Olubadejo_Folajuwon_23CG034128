import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Page Configuration
st.set_page_config(
    page_title="Wine Cultivar Prediction System",
    page_icon="🍷",
    layout="centered"
)

# Load Model and Scaler
@st.cache_resource
def load_models():
    try:
        model = joblib.load('model/wine_cultivar_model.pkl')
        scaler = joblib.load('model/scaler.pkl')
        return model, scaler
    except FileNotFoundError:
        st.error("Model files not found. Please ensure 'wine_cultivar_model.pkl' and 'scaler.pkl' are in the 'model' directory.")
        return None, None

model, scaler = load_models()

# Header
st.title("Wine Cultivar Prediction System")
st.markdown("**Name:** Olubadejo Folajuwon | **Matric Number:** 23CG034128")
st.markdown("---")
st.write("Enter the wine chemical properties below to predict its Cultivar (Origin).")

# Input Features
st.subheader("Chemical Properties")
col1, col2 = st.columns(2)

with col1:
    alcohol = st.number_input("Alcohol", min_value=10.0, max_value=15.0, value=13.0, step=0.1, format="%.2f")
    flavanoids = st.number_input("Flavanoids", min_value=0.0, max_value=6.0, value=2.0, step=0.1, format="%.2f")
    color_intensity = st.number_input("Color Intensity", min_value=1.0, max_value=13.0, value=5.0, step=0.1, format="%.2f")

with col2:
    hue = st.number_input("Hue", min_value=0.0, max_value=2.0, value=1.0, step=0.01, format="%.2f")
    proline = st.number_input("Proline", min_value=200, max_value=1700, value=750, step=10)
    magnesium = st.number_input("Magnesium", min_value=50, max_value=200, value=100, step=1)

# Prediction
if st.button("Predict Cultivar"):
    if model and scaler:
        # Prepare input array
        # Order must match training: alcohol, flavanoids, color_intensity, hue, proline, magnesium
        input_data = np.array([[alcohol, flavanoids, color_intensity, hue, proline, magnesium]])
        
        # Scale input
        input_scaled = scaler.transform(input_data)
        
        # Predict
        prediction = model.predict(input_scaled)
        prediction_prob = model.predict_proba(input_scaled)
        
        # Display Result
        st.markdown("---")
        
        # Target classes are 0, 1, 2. Let's map them to meaningful names if possible, or just Cultivar 1, 2, 3.
        # Sklearn Wine dataset target names: class_0, class_1, class_2
        cultivar_map = {0: "Class 0", 1: "Class 1", 2: "Class 2"}
        predicted_class = cultivar_map.get(prediction[0], f"Class {prediction[0]}")
        
        st.success(f"## Predicted Origin: {predicted_class}")
        
        # Display probabilities
        st.write("### Prediction Probabilities:")
        prob_df = pd.DataFrame(prediction_prob, columns=["Class 0", "Class 1", "Class 2"])
        st.dataframe(prob_df.style.format("{:.2%}"), use_container_width=True)
            
    else:
        st.error("Model could not be loaded.")

# Footer
st.markdown("---")
st.caption("Project 6 – Wine Cultivar Origin Prediction System")
