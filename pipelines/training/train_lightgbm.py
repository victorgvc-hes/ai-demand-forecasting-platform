from __future__ import annotations

import argparse
from pathlib import Path
import joblib

import numpy as np
import pandas as pd
import lightgbm as lgb


FEATURES = [
    "sell_price", "snap",
    "wday", "month", "year",
    "day_of_week", "day_of_month", "week_of_year",
    "has_event", "has_event_type",
    "lag_7", "lag_14", "lag_28",
    "rolling_mean_7", "rolling_std_7",
    "rolling_mean_28", "rolling_std_28",
]

TARGET = "demand"


def wape(y_true: pd.Series, y_pred: np.ndarray) -> float:
    denom = np.abs(y_true).sum()
    if denom == 0:
        return np.nan
    return np.abs(y_true - y_pred).sum() / denom


def bias(y_true: pd.Series, y_pred: np.ndarray) -> float:
    denom = np.abs(y_true).sum()
    if denom == 0:
        return np.nan
    return (y_pred - y_true).sum() / denom


def rmse(y_true: pd.Series, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def main(
    input_path: Path,
    output_metrics_path: Path,
    output_predictions_path: Path,
    output_model_path: Path,
    output_feature_importance_path: Path,
    horizon_days: int = 28,
) -> None:
    print(f"[INFO] Reading features parquet: {input_path}")
    df = pd.read_parquet(input_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["id", "date"]).reset_index(drop=True)

    max_date = df["date"].max()
    val_start = max_date - pd.Timedelta(days=horizon_days - 1)

    train_df = df[df["date"] < val_start].copy()
    val_df = df[df["date"] >= val_start].copy()

    print(f"[INFO] Train window: {train_df['date'].min().date()} -> {train_df['date'].max().date()}")
    print(f"[INFO] Validation window: {val_df['date'].min().date()} -> {val_df['date'].max().date()}")
    print(f"[INFO] Train rows: {len(train_df):,}")
    print(f"[INFO] Validation rows: {len(val_df):,}")

    X_train = train_df[FEATURES]
    y_train = train_df[TARGET]
    X_val = val_df[FEATURES]
    y_val = val_df[TARGET]

    model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        force_row_wise=True,
    )

    print("[INFO] Training LightGBM model...")
    model.fit(X_train, y_train)

    print("[INFO] Generating predictions...")
    val_df["prediction"] = model.predict(X_val)
    val_df["prediction"] = np.clip(val_df["prediction"], a_min=0, a_max=None)

    overall_wape = wape(y_val, val_df["prediction"].values)
    overall_bias = bias(y_val, val_df["prediction"].values)
    overall_rmse = rmse(y_val, val_df["prediction"].values)

    output_metrics_path.parent.mkdir(parents=True, exist_ok=True)
    output_predictions_path.parent.mkdir(parents=True, exist_ok=True)
    output_model_path.parent.mkdir(parents=True, exist_ok=True)
    output_feature_importance_path.parent.mkdir(parents=True, exist_ok=True)

    metrics_df = pd.DataFrame(
        [
            {
                "model": "lightgbm",
                "horizon_days": horizon_days,
                "train_start": str(train_df["date"].min().date()),
                "train_end": str(train_df["date"].max().date()),
                "val_start": str(val_df["date"].min().date()),
                "val_end": str(val_df["date"].max().date()),
                "wape": overall_wape,
                "bias": overall_bias,
                "rmse": overall_rmse,
                "n_train_rows": len(train_df),
                "n_val_rows": len(val_df),
            }
        ]
    )
    metrics_df.to_csv(output_metrics_path, index=False)

    pred_cols = ["id", "date", "demand", "prediction"]
    val_df[pred_cols].to_csv(output_predictions_path, index=False)

    joblib.dump(model, output_model_path)

    importance_df = pd.DataFrame(
        {
            "feature": FEATURES,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    importance_df.to_csv(output_feature_importance_path, index=False)

    print("[DONE] LightGBM training and evaluation completed successfully.")
    print(f"[METRICS] WAPE: {overall_wape:.4f}")
    print(f"[METRICS] Bias: {overall_bias:.4f}")
    print(f"[METRICS] RMSE: {overall_rmse:.4f}")
    print(f"[INFO] Metrics saved to: {output_metrics_path}")
    print(f"[INFO] Predictions saved to: {output_predictions_path}")
    print(f"[INFO] Model saved to: {output_model_path}")
    print(f"[INFO] Feature importances saved to: {output_feature_importance_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        default="data/processed/m5/training_features.parquet",
        help="Path to features parquet",
    )
    parser.add_argument(
        "--metrics_output",
        type=str,
        default="reports/lightgbm_metrics.csv",
        help="Path to metrics CSV",
    )
    parser.add_argument(
        "--predictions_output",
        type=str,
        default="reports/lightgbm_val_predictions.csv",
        help="Path to validation predictions CSV",
    )
    parser.add_argument(
        "--model_output",
        type=str,
        default="models/lightgbm_model.pkl",
        help="Path to serialized trained model",
    )
    parser.add_argument(
        "--feature_importance_output",
        type=str,
        default="reports/lightgbm_feature_importance.csv",
        help="Path to feature importance CSV",
    )
    parser.add_argument(
        "--horizon_days",
        type=int,
        default=28,
        help="Validation horizon in days",
    )
    args = parser.parse_args()

    main(
        input_path=Path(args.input),
        output_metrics_path=Path(args.metrics_output),
        output_predictions_path=Path(args.predictions_output),
        output_model_path=Path(args.model_output),
        output_feature_importance_path=Path(args.feature_importance_output),
        horizon_days=args.horizon_days,
    )