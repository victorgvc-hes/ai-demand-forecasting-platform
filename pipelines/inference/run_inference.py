from __future__ import annotations

import argparse
from pathlib import Path
import joblib
import pandas as pd
import numpy as np


FEATURES = [
    "sell_price", "snap",
    "wday", "month", "year",
    "day_of_week", "day_of_month", "week_of_year",
    "has_event", "has_event_type",
    "lag_7", "lag_14", "lag_28",
    "rolling_mean_7", "rolling_std_7",
    "rolling_mean_28", "rolling_std_28",
]


def main(
    features_path: Path,
    model_path: Path,
    output_path: Path
):

    print("[INFO] Loading model...")
    model = joblib.load(model_path)

    print("[INFO] Reading features dataset...")
    df = pd.read_parquet(features_path)

    X = df[FEATURES]

    print("[INFO] Generating predictions...")
    df["forecast"] = model.predict(X)

    df["forecast"] = np.clip(df["forecast"], a_min=0, a_max=None)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    df[["id","date","forecast"]].to_csv(output_path, index=False)

    print("[DONE] Inference completed successfully.")
    print(f"[INFO] Forecasts saved to: {output_path}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--features",
        type=str,
        default="data/processed/m5/training_features.parquet"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="models/lightgbm_model.pkl"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="reports/inference_forecasts.csv"
    )

    args = parser.parse_args()

    main(
        features_path=Path(args.features),
        model_path=Path(args.model),
        output_path=Path(args.output),
    )