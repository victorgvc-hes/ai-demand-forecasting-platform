# Model Comparison

## Baseline: Seasonal Naive
- WAPE: 0.8750
- Bias: -0.0410
- RMSE: 3.4031

## LightGBM
- WAPE: 0.6737
- Bias: -0.0629
- RMSE: 2.4564

## Improvement Summary
- WAPE improvement: 23.01%
- RMSE improvement: 27.82%

## Business Interpretation
The LightGBM model materially improves forecast accuracy versus the seasonal naive baseline. This suggests lower forecast error, which can translate into better replenishment decisions, lower stockout risk, and reduced inventory inefficiency.

A remaining opportunity is bias reduction, since the model still shows a mild underforecasting tendency.