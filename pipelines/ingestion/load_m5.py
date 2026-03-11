"""
M5 ingestion pipeline:
- Reads Kaggle M5 raw CSVs
- Validates schema + basic data quality
- Converts sales wide format (d_1..d_n) to long format
- Joins calendar + sell_prices
- Saves a clean dataset to data/processed/m5/demand_long.parquet

Expected raw files:
data/raw/m5/calendar.csv
data/raw/m5/sell_prices.csv
data/raw/m5/sales_train_evaluation.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd


REQUIRED_FILES = {
    "calendar": "calendar.csv",
    "sell_prices": "sell_prices.csv",
    "sales": "sales_train_evaluation.csv",
}


def _assert_file_exists(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")


def validate_calendar(df: pd.DataFrame) -> None:
    required_cols = {"d", "date", "wm_yr_wk"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"calendar.csv missing columns: {sorted(missing)}")

    if df["d"].isna().any():
        raise ValueError("calendar.csv has nulls in 'd'")

    if df["d"].duplicated().any():
        raise ValueError("calendar.csv has duplicated 'd' keys")

    # date should parse
    try:
        pd.to_datetime(df["date"])
    except Exception as e:
        raise ValueError(f"calendar.csv 'date' cannot be parsed: {e}") from e


def validate_prices(df: pd.DataFrame) -> None:
    required_cols = {"store_id", "item_id", "wm_yr_wk", "sell_price"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"sell_prices.csv missing columns: {sorted(missing)}")

    if df["sell_price"].isna().any():
        # prices can be missing for some weeks/items; we won't hard-fail,
        # but we will warn.
        pass

    if (df["sell_price"] < 0).any():
        raise ValueError("sell_prices.csv has negative sell_price values")


def validate_sales(df: pd.DataFrame) -> None:
    required_cols = {"id", "item_id", "dept_id", "cat_id", "store_id", "state_id"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"sales_train_evaluation.csv missing columns: {sorted(missing)}")

    d_cols = [c for c in df.columns if c.startswith("d_")]
    if len(d_cols) == 0:
        raise ValueError("sales_train_evaluation.csv has no day columns like d_1, d_2, ...")

    # demand should not be negative
    if (df[d_cols] < 0).any().any():
        raise ValueError("sales_train_evaluation.csv has negative demand values in d_* columns")


def build_long_sales(df_sales: pd.DataFrame) -> pd.DataFrame:
    id_cols = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    d_cols = [c for c in df_sales.columns if c.startswith("d_")]

    df_long = df_sales.melt(
        id_vars=id_cols,
        value_vars=d_cols,
        var_name="d",
        value_name="demand",
    )

    # enforce types
    df_long["demand"] = pd.to_numeric(df_long["demand"], errors="coerce").fillna(0).astype("int32")

    return df_long


def main(raw_dir: Path, out_path: Path) -> None:
    calendar_path = raw_dir / REQUIRED_FILES["calendar"]
    prices_path = raw_dir / REQUIRED_FILES["sell_prices"]
    sales_path = raw_dir / REQUIRED_FILES["sales"]

    _assert_file_exists(calendar_path)
    _assert_file_exists(prices_path)
    _assert_file_exists(sales_path)

    print(f"[INFO] Reading: {calendar_path}")
    df_cal = pd.read_csv(calendar_path)
    print(f"[INFO] Reading: {prices_path}")
    df_prices = pd.read_csv(prices_path)
    print(f"[INFO] Reading: {sales_path}")
    df_sales = pd.read_csv(sales_path)

    print("[INFO] Validating inputs...")
    validate_calendar(df_cal)
    validate_prices(df_prices)
    validate_sales(df_sales)

    print("[INFO] Building long sales table...")
    df_long = build_long_sales(df_sales)

    # calendar join
    print("[INFO] Joining calendar...")
    df_cal["date"] = pd.to_datetime(df_cal["date"])
    df_long = df_long.merge(
        df_cal[["d", "date", "wm_yr_wk", "wday", "month", "year", "event_name_1", "event_type_1", "snap_CA", "snap_TX", "snap_WI"]],
        on="d",
        how="left",
        validate="m:1",
    )

    # prices join (by store, item, week)
    print("[INFO] Joining sell_prices...")
    df_long = df_long.merge(
        df_prices[["store_id", "item_id", "wm_yr_wk", "sell_price"]],
        on=["store_id", "item_id", "wm_yr_wk"],
        how="left",
        validate="m:1",
    )

    # Basic post-join checks
    if df_long["date"].isna().any():
        raise ValueError("After join, some rows have null 'date' — check calendar keys")

    # Create SNAP column by state (optional but handy)
    # M5 provides snap_CA/snap_TX/snap_WI; map by state_id
    snap_map = {"CA": "snap_CA", "TX": "snap_TX", "WI": "snap_WI"}
    df_long["snap"] = 0
    for state, col in snap_map.items():
        mask = df_long["state_id"].eq(state)
        if col in df_long.columns:
            df_long.loc[mask, "snap"] = df_long.loc[mask, col].fillna(0).astype("int8")

    # Keep a clean set of columns (you can expand later)
    keep_cols = [
        "id", "item_id", "dept_id", "cat_id", "store_id", "state_id",
        "date", "d", "wm_yr_wk", "wday", "month", "year",
        "event_name_1", "event_type_1", "snap",
        "sell_price", "demand",
    ]
    df_out = df_long[keep_cols].sort_values(["id", "date"]).reset_index(drop=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Writing parquet: {out_path}")
    df_out.to_parquet(out_path, index=False)

    # Summary stats
    n_skus = df_out["id"].nunique()
    min_date = df_out["date"].min()
    max_date = df_out["date"].max()
    missing_price = df_out["sell_price"].isna().mean()

    print("[DONE] Output created successfully.")
    print(f"[STATS] rows={len(df_out):,} | skus={n_skus:,} | date_range={min_date.date()}..{max_date.date()}")
    print(f"[STATS] missing sell_price ratio = {missing_price:.2%} (expected for some weeks/items)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", type=str, default="data/raw/m5", help="Path to raw M5 CSV folder")
    parser.add_argument("--out", type=str, default="data/processed/m5/demand_long.parquet", help="Output parquet path")
    args = parser.parse_args()

    try:
        main(Path(args.raw_dir), Path(args.out))
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        raise