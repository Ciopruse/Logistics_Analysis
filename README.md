# Logistics Data Analysis

Python project covering all four milestones:

1. **Strategic Planning** — problem definition, KPIs, roadmap 
2. **Data Collection & Cleaning** — `scripts/generate_data.py`
3. **Exploratory Data Analysis & Visualization** — `scripts/clean_eda_model.py`
4. **Predictive Modeling & Optimization** — `scripts/clean_eda_model.py`

## Project Structure

```
├── scripts/
│   ├── generate_data.py       # Simulates the raw logistics dataset
│   └── clean_eda_model.py     # Cleaning, EDA, visualization, and predictive modeling
├── data/
│   ├── raw_logistics_data.csv
│   ├── cleaned_logistics_data.csv
│   ├── cleaning_report.json
│   ├── eda_summary.json
│   └── model_results.json
├── visuals/                   # All charts generated during EDA and modeling
└── requirements.txt
```

## What It Does

- Simulates a 1,200-record logistics shipment dataset (distance, weight, cost, delivery time, transport mode, carrier, region).
- Cleans the data: removes duplicates, imputes missing values (group-wise median / mode), caps outliers (IQR method), standardizes categorical text, and normalizes numeric features.
- Performs exploratory data analysis: descriptive statistics, correlation analysis, and 8 visualizations (distribution plots, boxplots, heatmap, regional/carrier comparisons, feature importance, predicted vs. actual).
- Trains and evaluates two regression models (Linear Regression, Random Forest) to predict delivery time, reporting RMSE, MAE, and R².
- Surfaces optimization recommendations based on feature importance (transport mode, carrier, and regional routing levers).

## How to Run

```bash
pip install -r requirements.txt
python scripts/generate_data.py
python scripts/clean_eda_model.py
```

Outputs (cleaned CSVs, JSON summaries, and PNG charts) are written to the working directory / `visuals/`.

