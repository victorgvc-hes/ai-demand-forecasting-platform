from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def wape(y_true: pd.Series, y_pred: pd.Series) -> float:
    denom = np.abs(y_true).sum()
    if denom == 0:
        return np.nan
    return np.abs(y_true - y_pred).sum() / denom


def bias(y_true: pd.Series, y_pred: pd.Series) -> float:
    denom = np.abs(y_true).sum()
    if denom == 0:
        return np.nan
    return (y_pred - y_true).sum() / denom


def rmse(y_true: pd.Series, y_pred: pd.Series) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def main(
    input_path: Path,
    output_path: Path,
    horizon_days: int = 28,
    seasonal_lag: int = 7,
    sample_n_ids: int | None = 1000,
) -> None:
    print(f"[INFO] Reading parquet: {input_path}")
    df = pd.read_parquet(input_path, columns=["id", "date", "demand"])
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["id", "date"]).reset_index(drop=True)

    if sample_n_ids is not None:
        ids = df["id"].drop_duplicates().head(sample_n_ids)
        df = df[df["id"].isin(ids)].copy()
        print(f"[INFO] Using sample of {len(ids):,} SKUs for faster baseline run")

    max_date = df["date"].max()
    val_start = max_date - pd.Timedelta(days=horizon_days - 1)

    print(f"[INFO] Validation window: {val_start.date()} -> {max_date.date()}")

    df["forecast"] = df.groupby("id")["demand"].shift(seasonal_lag)

    val_df = df[df["date"] >= val_start].copy()
    val_df = val_df.dropna(subset=["forecast"])

    val_df["forecast"] = val_df["forecast"].astype(float)

    overall_wape = wape(val_df["demand"], val_df["forecast"])
    overall_bias = bias(val_df["demand"], val_df["forecast"])
    overall_rmse = rmse(val_df["demand"], val_df["forecast"])

    per_sku = (
        val_df.groupby("id")
        .apply(
            lambda g: pd.Series(
                {
                    "wape": wape(g["demand"], g["forecast"]),
                    "bias": bias(g["demand"], g["forecast"]),
                    "rmse": rmse(g["demand"], g["forecast"]),
                    "n_obs": len(g),
                    "mean_demand": g["demand"].mean(),
                }
            )
        )
        .reset_index()
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    per_sku.to_csv(output_path, index=False)

    summary_path = output_path.parent / "baseline_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Seasonal Naive Baseline Summary\n")
        f.write(f"Input: {input_path}\n")
        f.write(f"Validation window: {val_start.date()} -> {max_date.date()}\n")
        f.write(f"Seasonal lag: {seasonal_lag}\n")
        f.write(f"Sample SKUs: {sample_n_ids if sample_n_ids is not None else 'ALL'}\n")
        f.write(f"Overall WAPE: {overall_wape:.4f}\n")
        f.write(f"Overall Bias: {overall_bias:.4f}\n")
        f.write(f"Overall RMSE: {overall_rmse:.4f}\n")

    print("[DONE] Baseline metrics created successfully.")
    print(f"[METRICS] Overall WAPE: {overall_wape:.4f}")
    print(f"[METRICS] Overall Bias: {overall_bias:.4f}")
    print(f"[METRICS] Overall RMSE: {overall_rmse:.4f}")
    print(f"[INFO] Per-SKU metrics saved to: {output_path}")
    print(f"[INFO] Summary saved to: {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        default="data/processed/m5/demand_long.parquet",
        help="Path to processed demand parquet",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="reports/baseline_metrics.csv",
        help="Path to output CSV with per-SKU metrics",
    )
    parser.add_argument(
        "--horizon_days",
        type=int,
        default=28,
        help="Validation horizon in days",
    )
    parser.add_argument(
        "--seasonal_lag",
        type=int,
        default=7,
        help="Seasonal lag for naive forecast",
    )
    parser.add_argument(
        "--sample_n_ids",
        type=int,
        default=1000,
        help="Number of SKUs to sample for faster runs; use 0 to run all",
    )
    args = parser.parse_args()

    sample_n_ids = None if args.sample_n_ids == 0 else args.sample_n_ids

    main(
        input_path=Path(args.input),
        output_path=Path(args.output),
        horizon_days=args.horizon_days,
        seasonal_lag=args.seasonal_lag,
        sample_n_ids=sample_n_ids,
    )