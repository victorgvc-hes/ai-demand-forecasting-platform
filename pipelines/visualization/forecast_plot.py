import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

INPUT_PATH = Path("reports/lightgbm_val_predictions.csv")
OUTPUT_PATH = Path("reports/forecast_vs_actual.png")


def main():

    print("[INFO] Reading validation predictions...")
    df = pd.read_csv(INPUT_PATH)

    df["date"] = pd.to_datetime(df["date"])

    # tomar un SKU para visualizar
    sample_sku = df["id"].iloc[0]
    df = df[df["id"] == sample_sku].sort_values("date")

    plt.figure(figsize=(10,5))

    plt.plot(df["date"], df["demand"], label="Actual Demand")
    plt.plot(df["date"], df["prediction"], label="Forecast")

    plt.title(f"Forecast vs Actual Demand — {sample_sku}")
    plt.xlabel("Date")
    plt.ylabel("Units")

    plt.legend()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH)

    print(f"[DONE] Plot saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()