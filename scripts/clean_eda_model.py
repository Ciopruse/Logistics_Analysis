import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import json

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 150

df = pd.read_csv("raw_logistics_data.csv", parse_dates=["shipment_date"])
n_raw = len(df)

# WEEK 2: CLEANING 
report = {}
report["raw_rows"] = n_raw
report["duplicates_found"] = int(df.duplicated(subset=[c for c in df.columns if c != "shipment_id"]).sum())

df["transport_mode"] = df["transport_mode"].str.strip().str.title()

df = df.drop_duplicates(subset="shipment_id", keep="first")
report["rows_after_dedup"] = len(df)

report["missing_before"] = df.isna().sum().to_dict()

df["carrier"] = df["carrier"].fillna(df["carrier"].mode()[0])

for col in ["weight_kg", "transportation_cost", "delivery_time_days"]:
    df[col] = df.groupby("transport_mode")[col].transform(lambda s: s.fillna(s.median()))

report["missing_after"] = df.isna().sum().to_dict()

def iqr_cap(series):
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    n_out = int(((series < lower) | (series > upper)).sum())
    return series.clip(lower, upper), n_out, lower, upper

df["transportation_cost"], n_out_cost, lo_c, hi_c = iqr_cap(df["transportation_cost"])
df["delivery_time_days"], n_out_time, lo_t, hi_t = iqr_cap(df["delivery_time_days"])
report["outliers_capped_cost"] = n_out_cost
report["outliers_capped_delivery_time"] = n_out_time
report["cost_bounds"] = [round(lo_c, 2), round(hi_c, 2)]
report["delivery_time_bounds"] = [round(lo_t, 2), round(hi_t, 2)]

num_cols = ["distance_km", "weight_kg", "transportation_cost", "delivery_time_days", "inventory_level", "shipment_volume_units"]
for c in num_cols:
    df[c + "_norm"] = (df[c] - df[c].min()) / (df[c].max() - df[c].min())

df.to_csv("cleaned_logistics_data.csv", index=False)
report["clean_rows"] = len(df)

with open("cleaning_report.json", "w") as f:
    json.dump(report, f, indent=2, default=str)
print(json.dumps(report, indent=2, default=str))

# WEEK 3: EDA & VISUALIZATION 
desc = df[num_cols].describe().T
desc.to_csv("eda_descriptive_stats.csv")

corr = df[num_cols].corr()
corr.to_csv("eda_correlation_matrix.csv")

plt.figure(figsize=(7, 4.5))
sns.histplot(df["delivery_time_days"], bins=30, kde=True, color="#d9622b")
plt.title("Distribution of Delivery Time (days)")
plt.xlabel("Delivery Time (days)")
plt.tight_layout()
plt.savefig("fig1_delivery_time_dist.png")
plt.close()

plt.figure(figsize=(7, 4.5))
sns.boxplot(data=df, x="transport_mode", y="transportation_cost", palette="Oranges")
plt.title("Transportation Cost by Mode")
plt.xlabel("Transport Mode")
plt.ylabel("Cost")
plt.tight_layout()
plt.savefig("fig2_cost_by_mode.png")
plt.close()

plt.figure(figsize=(6.5, 5.5))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdOrYl" if False else "coolwarm", center=0)
plt.title("Correlation Matrix of Key Logistics Variables")
plt.tight_layout()
plt.savefig("fig3_correlation_heatmap.png")
plt.close()

plt.figure(figsize=(7, 4.5))
region_avg = df.groupby("destination_region")["delivery_time_days"].mean().sort_values(ascending=False)
sns.barplot(x=region_avg.index, y=region_avg.values, color="#e08a3c")
plt.title("Average Delivery Time by Destination Region")
plt.ylabel("Avg Delivery Time (days)")
plt.xlabel("Region")
plt.tight_layout()
plt.savefig("fig4_region_delivery_time.png")
plt.close()

plt.figure(figsize=(7, 4.8))
sns.scatterplot(data=df, x="distance_km", y="delivery_time_days", hue="transport_mode", alpha=0.6, s=25, palette="Set2")
plt.title("Distance vs Delivery Time by Transport Mode")
plt.xlabel("Distance (km)")
plt.ylabel("Delivery Time (days)")
plt.tight_layout()
plt.savefig("fig5_distance_vs_time.png")
plt.close()

plt.figure(figsize=(7, 4.5))
delay_rate = df.groupby("carrier")["delayed"].mean().sort_values(ascending=False) * 100
sns.barplot(x=delay_rate.index, y=delay_rate.values, color="#c0472a")
plt.title("Delay Rate (%) by Carrier")
plt.ylabel("Delay Rate (%)")
plt.tight_layout()
plt.savefig("fig6_delay_rate_by_carrier.png")
plt.close()

eda_summary = {
    "mean_delivery_time": round(df["delivery_time_days"].mean(), 2),
    "median_delivery_time": round(df["delivery_time_days"].median(), 2),
    "std_delivery_time": round(df["delivery_time_days"].std(), 2),
    "mean_cost": round(df["transportation_cost"].mean(), 2),
    "overall_delay_rate_pct": round(df["delayed"].mean() * 100, 2),
    "cheapest_mode": df.groupby("transport_mode")["transportation_cost"].mean().idxmin(),
    "fastest_mode": df.groupby("transport_mode")["delivery_time_days"].mean().idxmin(),
    "worst_region_delivery": region_avg.index[0],
    "best_region_delivery": region_avg.index[-1],
    "worst_carrier_delay": delay_rate.index[0],
    "strongest_corr_with_delivery_time": corr["delivery_time_days"].drop("delivery_time_days").abs().idxmax(),
}
with open("eda_summary.json", "w") as f:
    json.dump(eda_summary, f, indent=2, default=str)
print(json.dumps(eda_summary, indent=2, default=str))

#WEEK 4: PREDICTIVE MODELING
features_num = ["distance_km", "weight_kg", "shipment_volume_units", "inventory_level"]
features_cat = ["transport_mode", "destination_region", "carrier"]
target = "delivery_time_days"

X = df[features_num + features_cat]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

preprocess = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), features_cat),
], remainder="passthrough")

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=300, max_depth=8, random_state=42),
}

model_results = {}
best_name, best_r2, best_pipe = None, -np.inf, None
for name, mdl in models.items():
    pipe = Pipeline([("prep", preprocess), ("model", mdl)])
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    mae = float(mean_absolute_error(y_test, preds))
    r2 = float(r2_score(y_test, preds))
    model_results[name] = {"RMSE": round(rmse, 3), "MAE": round(mae, 3), "R2": round(r2, 3)}
    if r2 > best_r2:
        best_name, best_r2, best_pipe = name, r2, pipe

with open("model_results.json", "w") as f:
    json.dump(model_results, f, indent=2)
print(json.dumps(model_results, indent=2))
print("Best model:", best_name)

rf_pipe = Pipeline([("prep", preprocess), ("model", RandomForestRegressor(n_estimators=300, max_depth=8, random_state=42))])
rf_pipe.fit(X_train, y_train)
ohe = rf_pipe.named_steps["prep"].named_transformers_["cat"]
cat_names = list(ohe.get_feature_names_out(features_cat))
all_feature_names = cat_names + features_num
importances = rf_pipe.named_steps["model"].feature_importances_
fi = pd.Series(importances, index=all_feature_names).sort_values(ascending=False).head(10)
fi.to_csv("feature_importance.csv")

plt.figure(figsize=(7, 5))
sns.barplot(x=fi.values, y=fi.index, color="#d9622b")
plt.title("Top 10 Feature Importances — Delivery Time Model")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig("fig7_feature_importance.png")
plt.close()

preds_best = best_pipe.predict(X_test)
plt.figure(figsize=(6, 6))
plt.scatter(y_test, preds_best, alpha=0.5, color="#d9622b", s=20)
lims = [min(y_test.min(), preds_best.min()), max(y_test.max(), preds_best.max())]
plt.plot(lims, lims, "k--", linewidth=1)
plt.xlabel("Actual Delivery Time (days)")
plt.ylabel("Predicted Delivery Time (days)")
plt.title(f"Predicted vs Actual — {best_name}")
plt.tight_layout()
plt.savefig("fig8_pred_vs_actual.png")
plt.close()

print("DONE")
