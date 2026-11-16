"""
==============================================================================
AI-Based Student Performance Risk Predictor - Production REST API
Filename: api.py
Framework: FastAPI + Pydantic v2 + Uvicorn
==============================================================================
"""

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
from typing import List, Dict
import numpy as np
import joblib
import tensorflow as tf
import os
import time

# Initialize FastAPI Application
app = FastAPI(
    title="Student Academic Performance Prediction API",
    description="REST API service delivering real-time neural network inference for student risk tier classification.",
    version="1.0.0"
)

# Global model and scaler holders
MODEL = None
SCALER = None
CLASS_NAMES = ["Low", "Medium", "High"]
RISK_LEVELS = ["High Risk", "Moderate Risk", "Low Risk"]

@app.on_event("startup")
def load_artifacts():
    """Load model and scaler into memory on service startup."""
    global MODEL, SCALER
    model_path = os.path.join("models", "student_performance.keras")
    scaler_path = os.path.join("models", "scaler.joblib")
    
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        MODEL = tf.keras.models.load_model(model_path)
        SCALER = joblib.load(scaler_path)
        print("[SERVICE] Model and scaler artifacts successfully initialized.")
    else:
        print("[SERVICE WARNING] Model artifacts not found yet. Train model first.")

# Pydantic Schemas for Strict Input Validation
class StudentFeaturePayload(BaseModel):
    age: int = Field(..., ge=15, le=60, description="Student chronological age in years")
    attendance_pct: float = Field(..., ge=0.0, le=100.0, description="Classroom attendance percentage")
    study_hours_per_week: float = Field(..., ge=0.0, le=70.0, description="Weekly independent study hours")
    previous_score: float = Field(..., ge=0.0, le=100.0, description="Previous term academic aggregate score")
    assignment_completion_pct: float = Field(..., ge=0.0, le=100.0, description="Percentage of submitted assignments")
    internal_assessment_score: float = Field(..., ge=0.0, le=100.0, description="Midterm internal assessment mark")
    participation_score: float = Field(..., ge=1.0, le=10.0, description="Classroom participation rating")
    sleep_hours_per_day: float = Field(..., ge=2.0, le=14.0, description="Average sleep hours per night")
    extracurricular_hours_per_week: float = Field(..., ge=0.0, le=40.0, description="Weekly extracurricular hours")
    failed_subjects: int = Field(..., ge=0, le=15, description="Number of cumulative failed courses")

    model_config = {
        "json_schema_extra": {
            "example": {
                "age": 20,
                "attendance_pct": 82.0,
                "study_hours_per_week": 16.5,
                "previous_score": 71.0,
                "assignment_completion_pct": 85.0,
                "internal_assessment_score": 68.0,
                "participation_score": 7.5,
                "sleep_hours_per_day": 7.0,
                "extracurricular_hours_per_week": 4.5,
                "failed_subjects": 0
            }
        }
    }

class PredictionResponse(BaseModel):
    predicted_category: str
    risk_level: str
    confidence_pct: float
    probabilities: Dict[str, float]
    latency_ms: float

class BatchPayload(BaseModel):
    students: List[StudentFeaturePayload]

class BatchResponse(BaseModel):
    total_processed: int
    predictions: List[PredictionResponse]

@app.get("/health", status_code=status.HTTP_200_OK)
def health_check():
    """Service health inspection endpoint."""
    if MODEL is None or SCALER is None:
        return {
            "status": "initializing",
            "service": "student-performance-ann",
            "model_loaded": False
        }
    return {
        "status": "healthy",
        "service": "student-performance-ann",
        "framework": "TensorFlow/Keras",
        "model_version": "1.0.0",
        "model_loaded": True
    }

@app.post("/predict", response_model=PredictionResponse, status_code=status.HTTP_200_OK)
def predict_single_student(student: StudentFeaturePayload):
    """Executes single student performance risk inference."""
    if MODEL is None or SCALER is None:
        raise HTTPException(status_code=503, detail="Model artifacts not yet loaded.")

    start_time = time.perf_counter()
    
    # 1. Arrange features in exact schema sequence
    input_vector = np.array([[
        student.age,
        student.attendance_pct,
        student.study_hours_per_week,
        student.previous_score,
        student.assignment_completion_pct,
        student.internal_assessment_score,
        student.participation_score,
        student.sleep_hours_per_day,
        student.extracurricular_hours_per_week,
        student.failed_subjects
    ]], dtype=np.float32)
    
    # 2. Standardize features
    scaled_vector = SCALER.transform(input_vector)
    
    # 3. Model forward pass
    probabilities = MODEL.predict(scaled_vector, verbose=0)[0]
    best_idx = int(np.argmax(probabilities))
    
    latency = (time.perf_counter() - start_time) * 1000.0
    
    return PredictionResponse(
        predicted_category=CLASS_NAMES[best_idx],
        risk_level=RISK_LEVELS[best_idx],
        confidence_pct=round(float(probabilities[best_idx]) * 100.0, 2),
        probabilities={
            "Low": round(float(probabilities[0]), 4),
            "Medium": round(float(probabilities[1]), 4),
            "High": round(float(probabilities[2]), 4)
        },
        latency_ms=round(latency, 2)
    )

@app.post("/predict/batch", response_model=BatchResponse, status_code=status.HTTP_200_OK)
def predict_batch_students(payload: BatchPayload):
    """Executes vectorized batch inference for multiple student records."""
    if MODEL is None or SCALER is None:
        raise HTTPException(status_code=503, detail="Model artifacts not yet loaded.")

    start_time = time.perf_counter()
    if len(payload.students) == 0:
        raise HTTPException(status_code=400, detail="Payload contains no student records.")
        
    matrix = np.array([[
        s.age, s.attendance_pct, s.study_hours_per_week, s.previous_score,
        s.assignment_completion_pct, s.internal_assessment_score,
        s.participation_score, s.sleep_hours_per_day,
        s.extracurricular_hours_per_week, s.failed_subjects
    ] for s in payload.students], dtype=np.float32)
    
    scaled_matrix = SCALER.transform(matrix)
    all_probabilities = MODEL.predict(scaled_matrix, verbose=0)
    best_indices = np.argmax(all_probabilities, axis=1)
    
    batch_results = []
    for i, probs in enumerate(all_probabilities):
        idx = int(best_indices[i])
        batch_results.append(PredictionResponse(
            predicted_category=CLASS_NAMES[idx],
            risk_level=RISK_LEVELS[idx],
            confidence_pct=round(float(probs[idx]) * 100.0, 2),
            probabilities={
                "Low": round(float(probs[0]), 4),
                "Medium": round(float(probs[1]), 4),
                "High": round(float(probs[2]), 4)
            },
            latency_ms=0.0
        ))
        
    total_time = (time.perf_counter() - start_time) * 1000.0
    for r in batch_results:
        r.latency_ms = round(total_time / len(batch_results), 2)
        
    return BatchResponse(
        total_processed=len(batch_results),
        predictions=batch_results
    )
