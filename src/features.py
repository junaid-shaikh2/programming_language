from __future__ import annotations

import pandas as pd


FEATURE_COLUMNS = [
    "return_1",
    "return_3",
    "return_5",
    "ma_5_gap",
    "ma_10_gap",
    "volatility_5",
    "high_low_spread",
    "momentum_3",
    "momentum_10",
]


def build_feature_frame(price_df: pd.DataFrame) -> pd.DataFrame:
    df = price_df.copy()
    df["return_1"] = df["close"].pct_change(1)
    df["return_3"] = df["close"].pct_change(3)
    df["return_5"] = df["close"].pct_change(5)
    df["ma_5"] = df["close"].rolling(5).mean()
    df["ma_10"] = df["close"].rolling(10).mean()
    df["ma_5_gap"] = (df["close"] / df["ma_5"]) - 1
    df["ma_10_gap"] = (df["close"] / df["ma_10"]) - 1
    df["volatility_5"] = df["return_1"].rolling(5).std()
    df["high_low_spread"] = (df["high"] - df["low"]) / df["close"]
    df["momentum_3"] = df["close"] - df["close"].shift(3)
    df["momentum_10"] = df["close"] - df["close"].shift(10)
    df["target"] = (df["close"].shift(-1) > df["close"]).astype(int)
    return df.dropna().copy()
