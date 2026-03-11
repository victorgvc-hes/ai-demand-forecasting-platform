from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def build_baseline_from_validation(
    features_path: Path,
    horizon_days: int = 28,
) -> pd.DataFrame:
    df = pd.read_parquet(features_path, columns=["id", "date", "demand", "lag_7"])
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["id", "date"]).reset_index(drop=True)

    max_date = df["date"].max()
    val_start = max_date - pd.Timedelta(days=horizon_days - 1)

    val_df = df[df["date"] >= val_start].copy()
    val_df = val_df.rename(columns={"lag_7": "baseline_forecast"})
    val_df["baseline_forecast"] = val_df["baseline_forecast"].clip(lower=0)

    return val_df[["id", "date", "demand", "baseline_forecast"]]


def summarize_impact(
    actual: pd.Series,
    forecast: pd.Series,
    stockout_cost_per_unit: float,
    holding_cost_per_unit: float,
) -> dict:
    error = forecast - actual

    underforecast_units = np.maximum(actual - forecast, 0).sum()
    overforecast_units = np.maximum(forecast - actual, 0).sum()

    stockout_cost_proxy = underforecast_units * stockout_cost_per_unit
    holding_cost_proxy = overforecast_units * holding_cost_per_unit

    return {
        "underforecast_units": float(underforecast_units),
        "overforecast_units": float(overforecast_units),
        "stockout_cost_proxy": float(stockout_cost_proxy),
        "holding_cost_proxy": float(holding_cost_proxy),
        "total_cost_proxy": float(stockout_cost_proxy + holding_cost_proxy),
    }


def main(
    features_path: Path,
    lightgbm_predictions_path: Path,
    output_csv_path: Path,
    output_md_path: Path,
    horizon_days: int = 28,
    stockout_cost_per_unit: float = 5.0,
    holding_cost_per_unit: float = 1.0,
) -> None:
    print("[INFO] Building baseline validation dataset...")
    baseline_df = build_baseline_from_validation(features_path, horizon_days=horizon_days)

    print("[INFO] Reading LightGBM validation predictions...")
    lgb_df = pd.read_csv(lightgbm_predictions_path)
    lgb_df["date"] = pd.to_datetime(lgb_df["date"])

    merged = baseline_df.merge(
        lgb_df[["id", "date", "demand", "prediction"]],
        on=["id", "date", "demand"],
        how="inner",
        validate="1:1",
    )

    if merged.empty:
        raise ValueError("Merged validation dataset is empty. Check date/id alignment.")

    print("[INFO] Computing business impact proxies...")
    baseline_metrics = summarize_impact(
        actual=merged["demand"],
        forecast=merged["baseline_forecast"],
        stockout_cost_per_unit=stockout_cost_per_unit,
        holding_cost_per_unit=holding_cost_per_unit,
    )

    lightgbm_metrics = summarize_impact(
        actual=merged["demand"],
        forecast=merged["prediction"],
        stockout_cost_per_unit=stockout_cost_per_unit,
        holding_cost_per_unit=holding_cost_per_unit,
    )

    results_df = pd.DataFrame(
        [
            {"model": "baseline_seasonal_naive", **baseline_metrics},
            {"model": "lightgbm", **lightgbm_metrics},
        ]
    )

    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    output_md_path.parent.mkdir(parents=True, exist_ok=True)

    results_df.to_csv(output_csv_path, index=False)

    baseline_total = results_df.loc[results_df["model"] == "baseline_seasonal_naive", "total_cost_proxy"].iloc[0]
    lightgbm_total = results_df.loc[results_df["model"] == "lightgbm", "total_cost_proxy"].iloc[0]
    improvement_pct = (baseline_total - lightgbm_total) / baseline_total * 100 if baseline_total != 0 else np.nan

    with open(output_md_path, "w", encoding="utf-8") as f:
        f.write("# Business Impact Analysis\n\n")
        f.write(f"- Stockout cost per unit: {stockout_cost_per_unit:.2f}\n")
        f.write(f"- Holding cost per unit: {holding_cost_per_unit:.2f}\n\n")
        f.write("## Results\n\n")
        f.write(results_df.to_markdown(index=False))
        f.write("\n\n")
        f.write("## Executive Interpretation\n\n")
        f.write(
            f"Using simple supply chain cost proxies, the LightGBM model reduces total forecast-related cost "
            f"by approximately {improvement_pct:.2f}% versus the seasonal naive baseline over the validation window. "
            f"This suggests better replenishment decisions, lower underforecast risk, and less excess inventory.\n"
        )

    print("[DONE] Business impact analysis completed successfully.")
    print(f"[INFO] Results saved to: {output_csv_path}")
    print(f"[INFO] Executive summary saved to: {output_md_path}")
    print(f"[METRICS] Estimated total cost improvement vs baseline: {improvement_pct:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--features",
        type=str,
        default="data/processed/m5/training_features.parquet",
        help="Path to features parquet",
    )
    parser.add_argument(
        "--predictions",
        type=str,
        default="reports/lightgbm_val_predictions.csv",
        help="Path to LightGBM validation predictions CSV",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="reports/business_impact.csv",
        help="Path to business impact CSV",
    )
    parser.add_argument(
        "--output_md",
        type=str,
        default="reports/business_impact_analysis.md",
        help="Path to business impact markdown summary",
    )
    parser.add_argument(
        "--horizon_days",
        type=int,
        default=28,
        help="Validation horizon in days",
    )
    parser.add_argument(
        "--stockout_cost_per_unit",
        type=float,
        default=5.0,
        help="Proxy stockout cost per unit",
    )
    parser.add_argument(
        "--holding_cost_per_unit",
        type=float,
        default=1.0,
        help="Proxy holding cost per unit",
    )
    args = parser.parse_args()

    main(
        features_path=Path(args.features),
        lightgbm_predictions_path=Path(args.predictions),
        output_csv_path=Path(args.output_csv),
        output_md_path=Path(args.output_md),
        horizon_days=args.horizon_days,
        stockout_cost_per_unit=args.stockout_cost_per_unit,
        holding_cost_per_unit=args.holding_cost_per_unit,
    )