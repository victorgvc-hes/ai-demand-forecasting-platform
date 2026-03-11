from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def add_lag_features(df: pd.DataFrame, group_col: str, target_col: str) -> pd.DataFrame:
    lag_days = [7, 14, 28]
    for lag in lag_days:
        df[f"lag_{lag}"] = df.groupby(group_col)[target_col].shift(lag)
    return df


def add_rolling_features(df: pd.DataFrame, group_col: str, target_col: str) -> pd.DataFrame:
    windows = [7, 28]
    for window in windows:
        shifted = df.groupby(group_col)[target_col].shift(1)
        df[f"rolling_mean_{window}"] = (
            shifted.groupby(df[group_col]).rolling(window).mean().reset_index(level=0, drop=True)
        )
        df[f"rolling_std_{window}"] = (
            shifted.groupby(df[group_col]).rolling(window).std().reset_index(level=0, drop=True)
        )
    return df


def main(input_path: Path, output_path: Path, sample_n_ids: int | None = 1000) -> None:
    print(f"[INFO] Reading parquet: {input_path}")
    cols = [
        "id", "date", "demand", "sell_price", "snap",
        "wday", "month", "year", "event_name_1", "event_type_1"
    ]
    df = pd.read_parquet(input_path, columns=cols)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["id", "date"]).reset_index(drop=True)

    if sample_n_ids is not None:
        ids = df["id"].drop_duplicates().head(sample_n_ids)
        df = df[df["id"].isin(ids)].copy()
        print(f"[INFO] Using sample of {len(ids):,} SKUs for faster feature build")

    print("[INFO] Creating calendar features...")
    df["day_of_week"] = df["date"].dt.dayofweek
    df["day_of_month"] = df["date"].dt.day
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype("int16")

    print("[INFO] Creating lag features...")
    df = add_lag_features(df, group_col="id", target_col="demand")

    print("[INFO] Creating rolling features...")
    df = add_rolling_features(df, group_col="id", target_col="demand")

    print("[INFO] Encoding simple event flags...")
    df["has_event"] = df["event_name_1"].notna().astype("int8")
    df["has_event_type"] = df["event_type_1"].notna().astype("int8")

    print("[INFO] Handling missing values...")
    df["sell_price"] = df.groupby("id")["sell_price"].ffill().bfill()
    df["rolling_std_7"] = df["rolling_std_7"].fillna(0)
    df["rolling_std_28"] = df["rolling_std_28"].fillna(0)

    feature_cols = [
        "id", "date", "demand",
        "sell_price", "snap",
        "wday", "month", "year",
        "day_of_week", "day_of_month", "week_of_year",
        "has_event", "has_event_type",
        "lag_7", "lag_14", "lag_28",
        "rolling_mean_7", "rolling_std_7",
        "rolling_mean_28", "rolling_std_28",
    ]

    df_out = df[feature_cols].copy()

    before_drop = len(df_out)
    df_out = df_out.dropna().reset_index(drop=True)
    after_drop = len(df_out)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Writing features parquet: {output_path}")
    df_out.to_parquet(output_path, index=False)

    print("[DONE] Training features created successfully.")
    print(f"[STATS] rows_before_dropna={before_drop:,}")
    print(f"[STATS] rows_after_dropna={after_drop:,}")
    print(f"[STATS] n_skus={df_out['id'].nunique():,}")
    print(f"[STATS] date_range={df_out['date'].min().date()}..{df_out['date'].max().date()}")


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
        default="data/processed/m5/training_features.parquet",
        help="Path to output features parquet",
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
        sample_n_ids=sample_n_ids,
    )