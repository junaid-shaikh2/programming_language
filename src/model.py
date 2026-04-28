from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score
from sklearn.model_selection import train_test_split

from src.features import FEATURE_COLUMNS, build_feature_frame


MODEL_DIR = Path("artifacts")
MODEL_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODEL_DIR / "forex_direction_model.joblib"


@dataclass
class TrainingResult:
    accuracy: float
    precision: float
    recall: float
    report: dict[str, Any]
    confusion_matrix: list[list[int]]
    backtest: dict[str, Any]
    latest_prediction: dict[str, Any]
    rows_used: int


def run_backtest(feature_df: pd.DataFrame, split_index: int) -> dict[str, Any]:
    test_df = feature_df.iloc[split_index:].copy()
    if len(test_df) < 20:
        return {
            "test_period_start": test_df.index[0].strftime("%Y-%m-%d"),
            "test_period_end": test_df.index[-1].strftime("%Y-%m-%d"),
            "strategy_return_pct": 0.0,
            "buy_and_hold_return_pct": 0.0,
            "winning_signal_rate": 0.0,
            "signals": [],
        }

    test_df["market_return"] = test_df["close"].pct_change().fillna(0.0)
    test_df["strategy_return"] = np.where(
        test_df["predicted_signal"].shift(1).fillna(0) == 1,
        test_df["market_return"],
        -test_df["market_return"],
    )
    test_df["strategy_curve"] = (1 + test_df["strategy_return"]).cumprod()
    test_df["buy_hold_curve"] = (1 + test_df["market_return"]).cumprod()
    test_df["signal_won"] = (
        ((test_df["predicted_signal"].shift(1).fillna(0) == 1) & (test_df["market_return"] > 0))
        | ((test_df["predicted_signal"].shift(1).fillna(0) == 0) & (test_df["market_return"] < 0))
    )

    return {
        "test_period_start": test_df.index[0].strftime("%Y-%m-%d"),
        "test_period_end": test_df.index[-1].strftime("%Y-%m-%d"),
        "strategy_return_pct": round(float((test_df["strategy_curve"].iloc[-1] - 1) * 100), 2),
        "buy_and_hold_return_pct": round(float((test_df["buy_hold_curve"].iloc[-1] - 1) * 100), 2),
        "winning_signal_rate": round(float(test_df["signal_won"].mean() * 100), 2),
        "signals": [
            {
                "date": index.strftime("%Y-%m-%d"),
                "predicted_signal": "UP" if int(row["predicted_signal"]) == 1 else "DOWN",
                "actual_direction": "UP" if int(row["target"]) == 1 else "DOWN",
                "market_return_pct": round(float(row["market_return"] * 100), 4),
                "strategy_return_pct": round(float(row["strategy_return"] * 100), 4),
            }
            for index, row in test_df.tail(30).iterrows()
        ],
    }


def train_model(price_df: pd.DataFrame) -> tuple[RandomForestClassifier, TrainingResult]:
    feature_df = build_feature_frame(price_df)
    X = feature_df[FEATURE_COLUMNS]
    y = feature_df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=3,
        random_state=42,
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    report = classification_report(y_test, predictions, output_dict=True, zero_division=0)
    matrix = confusion_matrix(y_test, predictions, labels=[0, 1]).tolist()

    feature_df.loc[X_test.index, "predicted_signal"] = predictions
    backtest = run_backtest(feature_df, split_index=len(X_train))

    latest_features = X.iloc[[-1]]
    probability_up = float(model.predict_proba(latest_features)[0][1])
    latest_prediction = {
        "prediction": "UP" if probability_up >= 0.5 else "DOWN",
        "probability_up": round(probability_up, 4),
        "probability_down": round(1 - probability_up, 4),
        "latest_close": round(float(feature_df.iloc[-1]["close"]), 5),
        "prediction_for_date": feature_df.index[-1].strftime("%Y-%m-%d"),
    }

    result = TrainingResult(
        accuracy=round(float(accuracy), 4),
        precision=round(float(precision), 4),
        recall=round(float(recall), 4),
        report=report,
        confusion_matrix=matrix,
        backtest=backtest,
        latest_prediction=latest_prediction,
        rows_used=len(feature_df),
    )
    return model, result


def save_model(model: RandomForestClassifier, metadata: dict[str, Any]) -> None:
    joblib.dump({"model": model, "metadata": metadata}, MODEL_PATH)


def load_model() -> dict[str, Any]:
    if not MODEL_PATH.exists():
        raise FileNotFoundError("No saved model found. Train the model first.")
    return joblib.load(MODEL_PATH)
