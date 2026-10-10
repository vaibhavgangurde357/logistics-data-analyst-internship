"""
Week 1 - Strategic Planning and Data Exploration in Logistics
Author: Vaibhav Nandlal Gangurde
Scenario: Last-mile delivery optimization for an e-commerce company (QuickKart, Pune)

This script simulates a delivery dataset, calculates the KPIs from the report,
clusters delivery zones and trains a baseline delivery-time model.
Run: pip install pandas numpy scikit-learn && python week1_kpi_and_model.py
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

rng = np.random.default_rng(42)
n = 2000

# ---- 1. Simulate a delivery dataset ----
df = pd.DataFrame({
    "Distance_km": rng.uniform(1, 25, n).round(2),
    "Traffic": rng.choice(["Low", "Medium", "High"], n, p=[0.3, 0.45, 0.25]),
    "Weather": rng.choice(["Clear", "Cloudy", "Rain"], n, p=[0.5, 0.3, 0.2]),
    "Vehicle_Type": rng.choice(["Bike", "Van"], n, p=[0.7, 0.3]),
    "hour": rng.integers(8, 22, n),
    "drop_latitude": rng.normal(18.52, 0.05, n),
    "drop_longitude": rng.normal(73.85, 0.05, n),
    "load_kg": rng.uniform(1, 40, n).round(1),
})
df["capacity_kg"] = np.where(df["Vehicle_Type"] == "Bike", 50, 150)
traffic_add = df["Traffic"].map({"Low": 0, "Medium": 8, "High": 20})
weather_add = df["Weather"].map({"Clear": 0, "Cloudy": 3, "Rain": 10})
df["Delivery_Time_min"] = (10 + 2.5 * df["Distance_km"] + traffic_add + weather_add
                           + rng.normal(0, 5, n)).round(1)
df["promised_time_min"] = 60
df["total_cost"] = (20 + 6 * df["Distance_km"]).round(2)

# ---- 2. KPIs ----
df["is_on_time"] = df["Delivery_Time_min"] <= df["promised_time_min"]
print("On-Time Delivery Rate : %.1f%%" % (df["is_on_time"].mean() * 100))
print("Avg Delivery Time     : %.1f min" % df["Delivery_Time_min"].mean())
print("Cost per Delivery     : Rs %.2f" % (df["total_cost"].sum() / len(df)))
print("Vehicle Utilization   : %.1f%%" % ((df["load_kg"] / df["capacity_kg"]).mean() * 100))

# ---- 3. Cluster delivery zones ----
coords = df[["drop_latitude", "drop_longitude"]]
df["zone"] = KMeans(n_clusters=8, random_state=42, n_init=10).fit_predict(coords)
print("\nAvg delivery time per zone:")
print(df.groupby("zone")["Delivery_Time_min"].mean().round(1).sort_values())

# ---- 4. Baseline delivery-time model ----
X = pd.get_dummies(df[["Distance_km", "Traffic", "Weather", "Vehicle_Type", "hour"]])
y = df["Delivery_Time_min"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=200, random_state=42).fit(X_tr, y_tr)
print("\nBaseline model MAE: %.2f min" % mean_absolute_error(y_te, model.predict(X_te)))
