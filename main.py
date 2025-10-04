import os
import streamlit as st
import pandas as pd
import pickle
from PIL import Image

# -----------------------------
# 1. Setup paths relative to main.py
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTCOME_MODEL_PATH = os.path.join(BASE_DIR, "models", "outcome_model.pkl")
DISEASE_MODEL_PATH = os.path.join(BASE_DIR, "models", "disease_model.pkl")
IMAGE_PATH = os.path.join(BASE_DIR, "assets", "healthcare.png")

# -----------------------------
# 2. Load models safely
# -----------------------------
def load_model(path):
    try:
        with open(path, "rb") as file:
            model = pickle.load(file)
        st.success(f"Loaded model: {os.path.basename(path)}")
        return model
    except FileNotFoundError:
        st.error(f"Model file not found at: {path}")
    except Exception as e:
        st.error(f"Error loading model {path}: {e}")
    return None

outcome_model = load_model(OUTCOME_MODEL_PATH)
disease_model = load_model(DISEASE_MODEL_PATH)

# -----------------------------
# 3. Sidebar for navigation
# -----------------------------
st.sidebar.title("Navigation")
selected_page = st.sidebar.radio("Go to", ["Home", "Prediction"])

# -----------------------------
# 4. Home Page
# -----------------------------
if selected_page == "Home":
    st.title("Disease Prediction App")
    st.markdown("""
    Welcome to the **Disease Prediction App**!  
    This app uses Machine Learning to predict whether you might have a disease based on your symptoms and other health indicators.
    """)

    if os.path.exists(IMAGE_PATH):
        image = Image.open(IMAGE_PATH)
        st.image(image, use_column_width=True)
    st.info("Navigate to the **Prediction** page using the sidebar to start.")

# -----------------------------
# 5. Prediction Page
# -----------------------------
elif selected_page == "Prediction":
    st.title("Prediction Form")
    st.markdown("### Please fill out the form below:")

    col1, col2 = st.columns(2)

    with col1:
        fever = st.selectbox("Fever", ["Yes", "No"])
        cough = st.selectbox("Cough", ["Yes", "No"])
        fatigue = st.selectbox("Fatigue", ["Yes", "No"])
        difficulty_breathing = st.selectbox("Difficulty Breathing", ["Yes", "No"])

    with col2:
        age = st.slider("Age", min_value=0, max_value=100, value=25)
        gender = st.selectbox("Gender", ["Male", "Female"])
        blood_pressure = st.selectbox("Blood Pressure", ["Low", "Normal", "High"])
        cholesterol = st.selectbox("Cholesterol Level", ["Normal", "High"])

    st.markdown("---")
    if st.button("Predict Disease"):
        if outcome_model is None or disease_model is None:
            st.warning("Models not loaded. Cannot make predictions.")
        else:
            input_data = pd.DataFrame({
                "Fever": [fever],
                "Cough": [cough],
                "Fatigue": [fatigue],
                "Difficulty Breathing": [difficulty_breathing],
                "Age": [age],
                "Gender": [gender],
                "Blood Pressure": [blood_pressure],
                "Cholesterol Level": [cholesterol]
            })

            try:
                outcome_prediction = outcome_model.predict(input_data)[0]
                st.markdown("### Prediction Result")
                if outcome_prediction == "Positive":
                    st.success("The predicted outcome is: **Positive**")
                    disease_prediction = disease_model.predict(input_data)[0]
                    st.write(f"The predicted disease is: **{disease_prediction}**")
                    st.warning("We recommend consulting with a healthcare professional.")
                else:
                    st.error("The predicted outcome is: **Negative**")
                    st.write("No disease predicted based on the provided inputs.")
            except Exception as e:
                st.error(f"Prediction failed: {e}")

# -----------------------------
# 6. Footer
# -----------------------------
st.markdown("---")
st.caption("Disease Prediction App © 2025")
