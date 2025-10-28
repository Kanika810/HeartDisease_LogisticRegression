import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt

# ---------- Page Config ----------
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- Load Model ----------
model = joblib.load("heart_disease_model.pkl")  # ensure your model file path

# ---------- Header Section ----------
st.markdown(
    """
    <style>
    .main-title {
        text-align: center;
        color: #e63946;
        font-size: 42px;
        font-weight: bold;
    }
    .sub-title {
        text-align: center;
        color: #555;
        font-size: 18px;
    }
    </style>
    """, unsafe_allow_html=True
)

st.markdown("<h1 class='main-title'>❤️ Heart Disease Risk Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-title'>Enter patient details to visualize risk and prediction insights</p>", unsafe_allow_html=True)
st.markdown("---")

# ---------- Input Section ----------
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=1, max_value=120, value=45)
    sex = st.selectbox("Sex", ["Male", "Female"])
    cp = st.selectbox("Chest Pain Type (0–3)", [0, 1, 2, 3])
    trestbps = st.number_input("Resting Blood Pressure", min_value=80, max_value=200, value=120)
    chol = st.number_input("Cholesterol", min_value=100, max_value=600, value=200)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])

with col2:
    restecg = st.selectbox("Resting ECG (0–2)", [0, 1, 2])
    thalach = st.number_input("Max Heart Rate Achieved", min_value=60, max_value=220, value=150)
    exang = st.selectbox("Exercise Induced Angina", [0, 1])
    oldpeak = st.number_input("ST Depression", min_value=0.0, max_value=10.0, value=1.0)
    slope = st.selectbox("Slope (0–2)", [0, 1, 2])
    ca = st.selectbox("Number of Major Vessels (0–3)", [0, 1, 2, 3])
    thal = st.selectbox("Thalassemia (1–3)", [1, 2, 3])

# ---------- Prediction ----------
if st.button("🔍 Predict"):
    sex_val = 1 if sex == "Male" else 0
    features = np.array([[age, sex_val, cp, trestbps, chol, fbs, restecg,
                          thalach, exang, oldpeak, slope, ca, thal]])

    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1] if hasattr(model, "predict_proba") else None

    # Display result
    if prediction == 1:
        st.markdown("<h3 style='text-align:center; color:#e63946;'>💔 High Risk of Heart Disease</h3>", unsafe_allow_html=True)
    else:
        st.markdown("<h3 style='text-align:center; color:#2a9d8f;'>💖 Low Risk (Healthy)</h3>", unsafe_allow_html=True)

    st.markdown("---")

    # ---------- Visualization Section ----------
    st.subheader("📊 Risk Probability Visualization")

    colA, colB = st.columns(2)

    with colA:
        if probability is not None:
            # Gauge-style chart
            fig, ax = plt.subplots(figsize=(3, 3))
            ax.barh(["Risk"], [probability], color="#e63946" if prediction == 1 else "#2a9d8f")
            ax.set_xlim(0, 1)
            ax.set_xlabel("Predicted Probability")
            st.pyplot(fig)

    with colB:
        # Feature visualization (basic demo)
        st.write("### Selected Health Factors")
        feature_labels = ["Age", "BP", "Cholesterol", "Heart Rate"]
        feature_values = [age, trestbps, chol, thalach]

        fig2, ax2 = plt.subplots()
        ax2.bar(feature_labels, feature_values, color="#457b9d")
        ax2.set_title("Patient Key Factors")
        st.pyplot(fig2)

# ---------- Sidebar ----------
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2966/2966486.png", width=120)
st.sidebar.markdown("### 👩‍⚕️ Developed by: Your Name")
st.sidebar.markdown("##### Project: Heart Disease Detection (Capstone)")
st.sidebar.markdown("---")
st.sidebar.write("🩺 This app predicts the likelihood of heart disease based on health indicators.")
