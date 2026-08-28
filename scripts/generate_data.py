import numpy as np
import pandas as pd

np.random.seed(42)

N = 1200
regions = ["North", "South", "East", "West", "Central"]
modes = ["Road", "Rail", "Air", "Sea"]
carriers = ["Carrier A", "Carrier B", "Carrier C", "Carrier D"]
warehouses = ["WH-Kolkata", "WH-Delhi", "WH-Mumbai", "WH-Chennai", "WH-Bengaluru"]

dates = pd.date_range("2025-01-01", "2025-12-31", periods=N)

distance_km = np.round(np.random.gamma(shape=3.0, scale=180, size=N), 1)
weight_kg = np.round(np.random.gamma(shape=2.5, scale=90, size=N), 1)
mode = np.random.choice(modes, size=N, p=[0.55, 0.2, 0.1, 0.15])

mode_speed = {"Road": 45, "Rail": 55, "Air": 550, "Sea": 25}
mode_base_cost_per_km = {"Road": 12, "Rail": 7, "Air": 55, "Sea": 4}

base_time = distance_km / np.array([mode_speed[m] for m in mode])
delivery_time_days = np.round(base_time / 24 + np.random.normal(0.4, 0.6, N).clip(min=-0.3), 1)
delivery_time_days = np.clip(delivery_time_days, 0.2, None)

transportation_cost = np.round(
    distance_km * np.array([mode_base_cost_per_km[m] for m in mode])
    + weight_kg * np.random.uniform(1.5, 3.5, N)
    + np.random.normal(0, 400, N),
    2,
)

inventory_level = np.random.randint(0, 5000, N)
shipment_volume = np.random.poisson(lam=120, size=N)

df = pd.DataFrame({
    "shipment_id": [f"SHP{100000+i}" for i in range(N)],
    "shipment_date": dates,
    "origin_warehouse": np.random.choice(warehouses, N),
    "destination_region": np.random.choice(regions, N),
    "carrier": np.random.choice(carriers, N),
    "transport_mode": mode,
    "distance_km": distance_km,
    "weight_kg": weight_kg,
    "shipment_volume_units": shipment_volume,
    "inventory_level": inventory_level,
    "transportation_cost": transportation_cost,
    "delivery_time_days": delivery_time_days,
})

threshold = df["delivery_time_days"].quantile(0.75)
df["delayed"] = ((df["delivery_time_days"] > threshold) & (np.random.rand(N) > 0.25)).astype(int)

rng = np.random.default_rng(7)

for col, frac in [("weight_kg", 0.04), ("transportation_cost", 0.03),
                   ("delivery_time_days", 0.02), ("carrier", 0.015)]:
    idx = rng.choice(N, size=int(N * frac), replace=False)
    df.loc[idx, col] = np.nan

out_idx = rng.choice(N, size=15, replace=False)
df.loc[out_idx, "transportation_cost"] *= rng.uniform(6, 10, size=15)
out_idx2 = rng.choice(N, size=10, replace=False)
df.loc[out_idx2, "delivery_time_days"] *= rng.uniform(5, 8, size=10)

dup_idx = rng.choice(N, size=20, replace=False)
df = pd.concat([df, df.loc[dup_idx]], ignore_index=True)

mask = rng.choice(df.index, size=40, replace=False)
df.loc[mask, "transport_mode"] = df.loc[mask, "transport_mode"].str.lower()

df = df.sample(frac=1, random_state=1).reset_index(drop=True)
df.to_csv("raw_logistics_data.csv", index=False)
print(df.shape)
print(df.isna().sum())
