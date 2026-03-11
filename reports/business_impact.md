# Business Impact Summary

## Forecasting performance
- Baseline WAPE: 0.8750
- LightGBM WAPE: 0.6737
- Baseline RMSE: 3.4031
- LightGBM RMSE: 2.4564

## Key interpretation
The LightGBM model improves accuracy materially versus the seasonal naive baseline. In supply chain terms, this can reduce forecast error, which should translate into fewer stockout situations, less avoidable inventory, and better replenishment decisions.

## Main business value drivers
- Better short-term replenishment signals
- Lower underforecast risk on volatile SKUs
- Reduced overforecast-driven excess inventory
- More stable operational planning

## Remaining limitation
The model still shows a mild underforecast bias, so bias reduction should be a next optimization target.