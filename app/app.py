
import os
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from tensorflow.keras.models import load_model

# Project app folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load model and scaler
model = load_model(os.path.join(BASE_DIR, "model.keras"))
target_scaler = joblib.load(
    os.path.join(BASE_DIR, "target_scaler.pkl")
)

# Create API
app = FastAPI(
    title="Beijing PM2.5 Prediction API",
    description="Predicts next-hour PM2.5 from the previous 24 hours.",
    version="1.0.0"
)

class PredictionRequest(BaseModel):
    sequence: list[list[float]] = Field(
        ...,
        description="24 hours × 15 preprocessed and scaled features"
    )

@app.get("/")
def home():
    return {"message": "Beijing PM2.5 Prediction API"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict")
def predict(request: PredictionRequest):
    data = np.asarray(request.sequence, dtype=float)

    if data.shape != (24, 15):
        raise HTTPException(
            status_code=400,
            detail="Input must contain exactly 24 rows and 15 features."
        )

    if not np.isfinite(data).all():
        raise HTTPException(
            status_code=400,
            detail="Input must contain only finite numeric values."
        )

    prediction_scaled = model.predict(
        data.reshape(1, 24, 15),
        verbose=0
    )

    prediction = target_scaler.inverse_transform(
        prediction_scaled.reshape(-1, 1)
    )[0, 0]

    return {
        "predicted_pm25": float(prediction),
        "unit": "µg/m³"
    }
