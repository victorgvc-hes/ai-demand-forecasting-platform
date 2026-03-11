# Case Study: AI Demand Forecasting Platform for Retail Supply Chains

## Problem

Retail supply chains depend heavily on accurate demand forecasts to plan replenishment, inventory levels, and distribution.  
Poor forecasts often lead to two costly outcomes:

- **Stockouts** (lost sales and poor customer experience)
- **Overstocking** (excess inventory and higher holding costs)

Traditional baseline forecasting methods (e.g., seasonal naive models) often fail to capture complex demand patterns driven by price changes, events, and short-term demand momentum.

The objective of this project was to build an **end-to-end AI-driven forecasting platform** capable of improving demand prediction accuracy while also quantifying the operational business impact.

---

## Solution

An end-to-end machine learning forecasting platform was designed and implemented using the **M5 retail dataset**, structured as a production-like system.

The system includes the following components:

### Data Engineering Pipeline
- Ingestion of raw retail transaction data
- Transformation and normalization into optimized Parquet datasets
- Structured dataset generation for machine learning

### Feature Engineering
Demand forecasting features were engineered to capture key business drivers:

- lag demand signals (7, 14, 28 days)
- rolling demand statistics
- calendar seasonality
- event indicators
- pricing effects

### Machine Learning Model

A **LightGBM regression model** was trained to predict product demand across multiple SKUs.

The model was benchmarked against a **seasonal naive baseline**.

### Model Serving

Two inference mechanisms were implemented:

- **Batch inference pipeline**
- **FastAPI real-time prediction API**

The API allows demand forecasts to be generated on demand using structured inputs.

---

## Results

Model performance improved significantly over the baseline.

| Model | WAPE | RMSE |
|------|------|------|
| Seasonal Naive Baseline | 0.8750 | 3.4031 |
| LightGBM | 0.6737 | 2.4564 |

Performance improvements:

- **~23% improvement in WAPE**
- **~27% improvement in RMSE**

Feature importance analysis showed the model primarily relied on:

- recent demand momentum
- rolling demand trends
- pricing signals
- calendar patterns

These drivers are consistent with real retail demand dynamics.

---

## Business Impact

To translate forecast accuracy into operational terms, a simplified cost proxy model was applied.

Assumptions:

- Stockout cost per unit = 5
- Holding cost per unit = 1

Results:

**Estimated operational cost reduction vs baseline: ~20.7%**

This improvement suggests that better demand forecasts can contribute to:

- reduced stockout risk
- lower excess inventory
- improved replenishment decisions
- more stable supply chain planning

---

## System Architecture
# Case Study: AI Demand Forecasting Platform for Retail Supply Chains

## Problem

Retail supply chains depend heavily on accurate demand forecasts to plan replenishment, inventory levels, and distribution.  
Poor forecasts often lead to two costly outcomes:

- **Stockouts** (lost sales and poor customer experience)
- **Overstocking** (excess inventory and higher holding costs)

Traditional baseline forecasting methods (e.g., seasonal naive models) often fail to capture complex demand patterns driven by price changes, events, and short-term demand momentum.

The objective of this project was to build an **end-to-end AI-driven forecasting platform** capable of improving demand prediction accuracy while also quantifying the operational business impact.

---

## Solution

An end-to-end machine learning forecasting platform was designed and implemented using the **M5 retail dataset**, structured as a production-like system.

The system includes the following components:

### Data Engineering Pipeline
- Ingestion of raw retail transaction data
- Transformation and normalization into optimized Parquet datasets
- Structured dataset generation for machine learning

### Feature Engineering
Demand forecasting features were engineered to capture key business drivers:

- lag demand signals (7, 14, 28 days)
- rolling demand statistics
- calendar seasonality
- event indicators
- pricing effects

### Machine Learning Model

A **LightGBM regression model** was trained to predict product demand across multiple SKUs.

The model was benchmarked against a **seasonal naive baseline**.

### Model Serving

Two inference mechanisms were implemented:

- **Batch inference pipeline**
- **FastAPI real-time prediction API**

The API allows demand forecasts to be generated on demand using structured inputs.

---

## Results

Model performance improved significantly over the baseline.

| Model | WAPE | RMSE |
|------|------|------|
| Seasonal Naive Baseline | 0.8750 | 3.4031 |
| LightGBM | 0.6737 | 2.4564 |

Performance improvements:

- **~23% improvement in WAPE**
- **~27% improvement in RMSE**

Feature importance analysis showed the model primarily relied on:

- recent demand momentum
- rolling demand trends
- pricing signals
- calendar patterns

These drivers are consistent with real retail demand dynamics.

---

## Business Impact

To translate forecast accuracy into operational terms, a simplified cost proxy model was applied.

Assumptions:

- Stockout cost per unit = 5
- Holding cost per unit = 1

Results:

**Estimated operational cost reduction vs baseline: ~20.7%**

This improvement suggests that better demand forecasts can contribute to:

- reduced stockout risk
- lower excess inventory
- improved replenishment decisions
- more stable supply chain planning

---

## System Architecture
Raw CSV Data
↓
Data Ingestion Pipeline
↓
Feature Engineering
↓
Model Training (LightGBM)
↓
Model Artifact (.pkl)
↓
Batch Inference Pipeline
↓
FastAPI Prediction Service
↓
Business Impact Evaluation

---

## Technologies Used

- Python
- Pandas / NumPy
- LightGBM
- FastAPI
- Matplotlib
- Parquet data pipelines

---

## Key Takeaways

This project demonstrates how machine learning can be integrated into a production-style supply chain forecasting workflow, connecting:

- **data engineering**
- **machine learning**
- **API deployment**
- **business impact evaluation**

The result is a forecasting system that not only predicts demand but also quantifies its potential operational value.
