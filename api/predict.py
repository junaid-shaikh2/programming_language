from __future__ import annotations

from fastapi import FastAPI, HTTPException, Query

from src.data_sources import APIError
from src.service import run_training_pipeline


app = FastAPI(title="Forex Direction Predictor API")


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "Forex prediction API is running."}


@app.get("/api/predict")
def predict(
    base: str = Query(..., min_length=3, max_length=3),
    quote: str = Query(..., min_length=3, max_length=3),
):
    try:
        return run_training_pipeline(base.upper(), quote.upper())
    except APIError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
