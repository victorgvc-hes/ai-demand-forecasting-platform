# Business Impact Analysis

- Stockout cost per unit: 5.00
- Holding cost per unit: 1.00

## Results

| model                   |   underforecast_units |   overforecast_units |   stockout_cost_proxy |   holding_cost_proxy |   total_cost_proxy |
|:------------------------|----------------------:|---------------------:|----------------------:|---------------------:|-------------------:|
| baseline_seasonal_naive |               24893   |              22663   |                124465 |              22663   |             147128 |
| lightgbm                |               20018.3 |              16598.1 |                100092 |              16598.1 |             116690 |

## Executive Interpretation

Using simple supply chain cost proxies, the LightGBM model reduces total forecast-related cost by approximately 20.69% versus the seasonal naive baseline over the validation window. This suggests better replenishment decisions, lower underforecast risk, and less excess inventory.
