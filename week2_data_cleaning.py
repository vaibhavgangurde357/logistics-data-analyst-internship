"""
Week 2 - Data Collection, Cleaning and Preprocessing for Logistics Analysis
Author: Vaibhav Nandlal Gangurde

Simulates a raw (messy) delivery dataset similar in structure to public datasets
such as the Amazon Delivery Dataset (Kaggle), then cleans and preprocesses it.
Run: pip install pandas numpy scikit-learn && python week2_data_cleaning.py
"""
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

rng = np.random.default_rng(7)
n = 3000

# >>> collect
# ---- Step 1: Data collection (simulated raw data with real-world quality issues) ----
raw = pd.DataFrame({
    "Order_ID": [f"ORD{100000 + i}" for i in range(n)],
    "Order_Date": pd.date_range("2026-01-01", periods=n, freq="45min").astype(str),
    "Distance_km": rng.uniform(1, 25, n).round(2),
    "Traffic": rng.choice(["Low", "Medium", "High"], n, p=[0.3, 0.45, 0.25]),
    "Weather": rng.choice(["Clear", "Cloudy", "Rain"], n, p=[0.5, 0.3, 0.2]),
    "Vehicle_Type": rng.choice(["Bike", "Van"], n, p=[0.7, 0.3]),
    "Agent_Age": rng.integers(20, 55, n).astype(float),
    "Agent_Rating": rng.uniform(3.5, 5.0, n).round(1),
    "hour": rng.integers(8, 22, n),
    "load_kg": rng.uniform(1, 40, n).round(1),
    "drop_latitude": rng.normal(18.52, 0.05, n),
    "drop_longitude": rng.normal(73.85, 0.05, n),
})
t_add = raw["Traffic"].map({"Low": 0, "Medium": 8, "High": 20})
w_add = raw["Weather"].map({"Clear": 0, "Cloudy": 3, "Rain": 10})
peak = raw["hour"].isin([9, 10, 18, 19, 20]) * 6
raw["Delivery_Time_min"] = (10 + 2.5 * raw["Distance_km"] + t_add + w_add + peak
                            + rng.normal(0, 5, n)).round(1)
raw["Shipping_Cost"] = (20 + 6 * raw["Distance_km"] + 0.5 * raw["load_kg"]
                        + rng.normal(0, 8, n)).round(2)

# inject data quality problems
raw.loc[rng.choice(n, 150, replace=False), "Agent_Rating"] = np.nan      # missing
raw.loc[rng.choice(n, 90, replace=False), "Weather"] = np.nan            # missing
raw.loc[rng.choice(n, 60, replace=False), "Delivery_Time_min"] = np.nan  # missing
raw.loc[rng.choice(n, 200, replace=False), "Traffic"] = rng.choice(
    ["high ", "HIGH", "low", " Medium", "medium"], 200)                  # inconsistent text
raw.loc[rng.choice(n, 25, replace=False), "Distance_km"] = rng.uniform(150, 600, 25).round(1)   # outliers
raw.loc[rng.choice(n, 20, replace=False), "Delivery_Time_min"] = rng.uniform(400, 900, 20).round(1)
raw.loc[rng.choice(n, 15, replace=False), "Agent_Age"] = rng.choice([5, 12, 150, 999], 15)      # impossible
raw.loc[rng.choice(n, 10, replace=False), "Shipping_Cost"] = -rng.uniform(10, 50, 10).round(2)  # negative
raw = pd.concat([raw, raw.sample(80, random_state=1)], ignore_index=True)                       # duplicates
raw.to_csv("logistics_raw.csv", index=False)
# <<< collect

report = {}
report["raw_shape"] = list(raw.shape)

# >>> inspect
# ---- Step 2: Inspect data quality ----
df = pd.read_csv("logistics_raw.csv")
missing = df.isnull().sum()
print("Missing values:\n", missing[missing > 0])
print("Duplicate rows:", df.duplicated().sum())
print("Traffic categories:", df["Traffic"].unique())
print(df[["Distance_km", "Agent_Age", "Delivery_Time_min", "Shipping_Cost"]].describe().round(1))
# <<< inspect
report["missing"] = {k: int(v) for k, v in missing[missing > 0].items()}
report["duplicates"] = int(df.duplicated().sum())
report["traffic_raw"] = sorted(map(str, df["Traffic"].unique()))
report["out_distance_gt100"] = int((df["Distance_km"] > 100).sum())
report["out_time_gt300"] = int((df["Delivery_Time_min"] > 300).sum())
report["bad_age"] = int(((df["Agent_Age"] < 18) | (df["Agent_Age"] > 70)).sum())
report["neg_cost"] = int((df["Shipping_Cost"] < 0).sum())

# >>> clean
# ---- Step 3: Cleaning ----
# 3.1 Remove duplicate orders
df = df.drop_duplicates()

# 3.2 Fix data types and inconsistent text
df["Order_Date"] = pd.to_datetime(df["Order_Date"])
for col in ["Traffic", "Weather", "Vehicle_Type"]:
    df[col] = df[col].str.strip().str.title()

# 3.3 Treat impossible values as missing
df.loc[(df["Agent_Age"] < 18) | (df["Agent_Age"] > 70), "Agent_Age"] = np.nan
df.loc[df["Shipping_Cost"] < 0, "Shipping_Cost"] = np.nan
df.loc[df["Distance_km"] > 100, "Distance_km"] = np.nan   # city delivery cannot exceed 100 km

# 3.4 Outlier detection with the IQR rule on delivery time
q1, q3 = df["Delivery_Time_min"].quantile([0.25, 0.75])
iqr = q3 - q1
low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
n_out = ((df["Delivery_Time_min"] < low) | (df["Delivery_Time_min"] > high)).sum()
df.loc[(df["Delivery_Time_min"] < low) | (df["Delivery_Time_min"] > high), "Delivery_Time_min"] = np.nan

# 3.5 Handle missing values: median for numeric, mode for categorical
num_cols = ["Distance_km", "Agent_Age", "Agent_Rating", "Delivery_Time_min", "Shipping_Cost"]
for col in num_cols:
    df[col] = df[col].fillna(df[col].median())
for col in ["Traffic", "Weather"]:
    df[col] = df[col].fillna(df[col].mode()[0])
print("Remaining missing values:", df.isnull().sum().sum())
# <<< clean
report["iqr_bounds"] = [round(float(low), 1), round(float(high), 1)]
report["iqr_outliers"] = int(n_out)
report["clean_shape"] = list(df.shape)
report["remaining_missing"] = int(df.isnull().sum().sum())
report["traffic_clean"] = sorted(df["Traffic"].unique())

# >>> normalize
# ---- Step 4: Feature engineering and normalization ----
df["is_peak_hour"] = df["hour"].isin([9, 10, 18, 19, 20]).astype(int)
df["weekday"] = df["Order_Date"].dt.day_name()
df["is_late"] = (df["Delivery_Time_min"] > 60).astype(int)
df.to_csv("logistics_clean.csv", index=False)       # unscaled copy for EDA (Week 3)

scale_cols = ["Distance_km", "Agent_Age", "Agent_Rating", "load_kg", "Shipping_Cost"]
minmax = df.copy()
minmax[scale_cols] = MinMaxScaler().fit_transform(df[scale_cols])        # range 0 to 1
zscore = df.copy()
zscore[scale_cols] = StandardScaler().fit_transform(df[scale_cols])      # mean 0, std 1
model_ready = pd.get_dummies(minmax, columns=["Traffic", "Weather", "Vehicle_Type"], drop_first=True)
model_ready.to_csv("logistics_model_ready.csv", index=False)
# <<< normalize
report["minmax_range"] = [round(float(minmax["Distance_km"].min()), 2), round(float(minmax["Distance_km"].max()), 2)]
report["zscore_mean"] = round(float(zscore["Distance_km"].mean()), 2)
report["zscore_std"] = round(float(zscore["Distance_km"].std()), 2)
report["model_ready_cols"] = int(model_ready.shape[1])
report["time_mean_before"] = round(float(pd.read_csv("logistics_raw.csv")["Delivery_Time_min"].mean()), 1)
report["time_mean_after"] = round(float(df["Delivery_Time_min"].mean()), 1)
report["dist_max_after"] = round(float(df["Distance_km"].max()), 1)
print("Clean shape:", df.shape, "| Model-ready columns:", model_ready.shape[1])
json.dump(report, open("week2_results.json", "w"), indent=1)
