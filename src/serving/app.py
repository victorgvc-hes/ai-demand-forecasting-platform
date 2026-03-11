from __future__ import annotations

from pathlib import Path
import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel


MODEL_PATH = Path("models/lightgbm_model.pkl")

FEATURES = [
    "sell_price", "snap",
    "wday", "month", "year",
    "day_of_week", "day_of_month", "week_of_year",
    "has_event", "has_event_type",
    "lag_7", "lag_14", "lag_28",
    "rolling_mean_7", "rolling_std_7",
    "rolling_mean_28", "rolling_std_28",
]


class ForecastRequest(BaseModel):
    sell_price: float
    snap: int
    wday: int
    month: int
    year: int
    day_of_week: int
    day_of_month: int
    week_of_year: int
    has_event: int
    has_event_type: int
    lag_7: float
    lag_14: float
    lag_28: float
    rolling_mean_7: float
    rolling_std_7: float
    rolling_mean_28: float
    rolling_std_28: float


app = FastAPI(
    title="AI Demand Forecasting API",
    description="FastAPI service for demand forecasting using a trained LightGBM model",
    version="1.0.0",
)

model = joblib.load(MODEL_PATH)


@app.get("/")
def root():
    return {
        "message": "AI Demand Forecasting API is running",
        "model_path": str(MODEL_PATH),
        "n_features": len(FEATURES),
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(request: ForecastRequest):
    values = [[getattr(request, feature) for feature in FEATURES]]
    prediction = model.predict(values)[0]
    prediction = float(np.clip(prediction, a_min=0, a_max=None))

    return {
        "forecast": prediction
    }