"""
==============================================================================
AI-Based Student Performance Risk Predictor - Interactive Web Application
Filename: app.py
Framework: Streamlit
==============================================================================
"""

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
import os

# Page configuration
st.set_page_config(
    page_title="Student Performance Risk Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
    .risk-high { color: #d9534f; font-weight: bold; }
    .risk-med { color: #f0ad4e; font-weight: bold; }
    .risk-low { color: #5cb85c; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_artifacts():
    """Load and cache model and scaler artifacts."""
    model_path = os.path.join("models", "student_performance.keras")
    scaler_path = os.path.join("models", "scaler.joblib")
    
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        st.error("Artifacts not found in models/ directory. Run training script first.")
        st.stop()
        
    model = tf.keras.models.load_model(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

# Header
st.title("🎓 Student Academic Performance Risk Predictor")
st.markdown("""
This system evaluates multi-dimensional student engagement metrics using an Artificial Neural Network 
to estimate academic performance tiers (**Low**, **Medium**, or **High**) and trigger proactive interventions.
""")
st.divider()

# Sidebar: Student Input Form
st.sidebar.header("📋 Student Profile Inputs")

with st.sidebar.form(key="student_input_form"):
    st.subheader("Academic History")
    previous_score = st.slider("Previous Score (0 - 100)", min_value=0.0, max_value=100.0, value=65.0, step=0.5)
    internal_assessment = st.slider("Internal Assessment (0 - 100)", min_value=0.0, max_value=100.0, value=60.0, step=0.5)
    failed_subjects = st.number_input("Prior Failed Subjects / Backlogs", min_value=0, max_value=10, value=0, step=1)
    
    st.subheader("Semester Engagement")
    attendance_pct = st.slider("Classroom Attendance (%)", min_value=0.0, max_value=100.0, value=75.0, step=0.5)
    study_hours = st.slider("Study Hours / Week", min_value=0.0, max_value=50.0, value=15.0, step=0.5)
    assignment_pct = st.slider("Assignment Completion (%)", min_value=0.0, max_value=100.0, value=75.0, step=0.5)
    participation = st.slider("Participation Rating (1.0 - 10.0)", min_value=1.0, max_value=10.0, value=6.0, step=0.1)
    
    st.subheader("Demographics & Lifestyle")
    age = st.number_input("Age", min_value=16, max_value=40, value=20, step=1)
    sleep_hours = st.slider("Sleep Hours / Day", min_value=2.0, max_value=12.0, value=7.0, step=0.25)
    extracurricular = st.slider("Extracurricular Hours / Week", min_value=0.0, max_value=30.0, value=5.0, step=0.5)
    
    submit_button = st.form_submit_button(label="Predict Academic Standing")

# Prediction Processing
if submit_button:
    try:
        model, scaler = load_artifacts()
    except Exception as e:
        st.error(f"Error loading model artifacts: {e}")
        st.stop()

    raw_features = np.array([[
        age,
        attendance_pct,
        study_hours,
        previous_score,
        assignment_pct,
        internal_assessment,
        participation,
        sleep_hours,
        extracurricular,
        failed_subjects
    ]], dtype=np.float32)
    
    # Preprocessing transformation
    scaled_features = scaler.transform(raw_features)
    
    # Neural Network Forward Pass
    probabilities = model.predict(scaled_features, verbose=0)[0]
    predicted_class_idx = np.argmax(probabilities)
    confidence = probabilities[predicted_class_idx] * 100
    
    classes = ["Low Performance", "Medium Performance", "High Performance"]
    risk_levels = ["High Academic Risk", "Moderate Academic Risk", "Low Academic Risk"]
    predicted_label = classes[predicted_class_idx]
    predicted_risk = risk_levels[predicted_class_idx]
    
    # Output Layout
    col1, col2, col3 = st.columns([1, 1, 1.2])
    
    with col1:
        st.subheader("Model Classification")
        if predicted_class_idx == 0:
            st.error(f"### {predicted_label}")
            st.markdown(f"**Institutional Status:** <span class='risk-high'>{predicted_risk}</span>", unsafe_allow_html=True)
        elif predicted_class_idx == 1:
            st.warning(f"### {predicted_label}")
            st.markdown(f"**Institutional Status:** <span class='risk-med'>{predicted_risk}</span>", unsafe_allow_html=True)
        else:
            st.success(f"### {predicted_label}")
            st.markdown(f"**Institutional Status:** <span class='risk-low'>{predicted_risk}</span>", unsafe_allow_html=True)
        st.metric("Model Confidence", f"{confidence:.1f}%")
        
    with col2:
        st.subheader("Probability Distribution")
        prob_df = pd.DataFrame({
            "Category": ["Low", "Medium", "High"],
            "Probability": probabilities
        })
        st.bar_chart(prob_df.set_index("Category"))
        
    with col3:
        st.subheader("Prescriptive Action Plan")
        if predicted_class_idx == 0:
            st.markdown("""
            * **Immediate Action:** Assign designated faculty mentor within 48 hours.
            * **Curriculum Adjustment:** Mandate 4 hours of weekly remedial tutoring in core subjects.
            * **Attendance Monitoring:** Establish weekly attendance check-ins with academic advisory.
            * **Workload Guidance:** Temporarily cap extracurricular commitments until assessment scores rebound.
            """)
        elif predicted_class_idx == 1:
            st.markdown("""
            * **Proactive Mentorship:** Encourage bi-weekly check-ins with course teaching assistants.
            * **Targeted Support:** Recommend participation in peer-led study groups for upcoming midterms.
            * **Habit Formation:** Suggest increasing dedicated study time by 3-5 hours weekly.
            """)
        else:
            st.markdown("""
            * **Advanced Placement:** Encourage enrollment in honors seminars or independent undergraduate research.
            * **Leadership Role:** Nominate for peer tutoring and department leadership opportunities.
            * **Continuity:** Maintain current schedule balance and academic workload.
            """)
            
    st.divider()
    
    # Feature Context Table
    st.subheader("Processed Feature Vector Context")
    feature_names = [
        "Age", "Attendance %", "Study Hours", "Previous Score", 
        "Assignment %", "Internal Assessment", "Participation", 
        "Sleep Hours", "Extracurricular Hours", "Failed Subjects"
    ]
    summary_df = pd.DataFrame({
        "Feature": feature_names,
        "Raw Input Value": raw_features[0],
        "Standardized Value (Z-Score)": scaled_features[0]
    })
    st.dataframe(summary_df.T, use_container_width=True)
else:
    st.info("👈 Enter student parameters in the left sidebar and click 'Predict Academic Standing' to run the evaluation.")
