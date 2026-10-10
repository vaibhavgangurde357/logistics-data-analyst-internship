"""
Week 3 - Advanced Data Analysis and Visualization in Logistics
Author: Vaibhav Nandlal Gangurde

Input : logistics_clean.csv (created by week2_data_cleaning.py)
Output: charts/*.png and week3_results.json
Run   : pip install pandas numpy matplotlib seaborn && python week3_eda_visualization.py
"""
import json
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", palette="Blues_d")
df = pd.read_csv("logistics_clean.csv", parse_dates=["Order_Date"])
os.makedirs("charts", exist_ok=True)
res = {}

# >>> eda
# ---- Step 1: Exploratory Data Analysis ----
num = ["Distance_km", "Delivery_Time_min", "Shipping_Cost", "load_kg", "Agent_Rating"]
print(df[num].describe().round(2))
print("Skewness:\n", df[num].skew().round(2))
corr = df[num].corr().round(2)
print(corr)
print("Late-delivery rate: %.1f%%" % (df["is_late"].mean() * 100))
# <<< eda
res["describe"] = df[num].describe().round(2).to_dict()
res["median_time"] = round(float(df["Delivery_Time_min"].median()), 1)
res["mean_time"] = round(float(df["Delivery_Time_min"].mean()), 1)
res["std_time"] = round(float(df["Delivery_Time_min"].std()), 1)
res["skew_time"] = round(float(df["Delivery_Time_min"].skew()), 2)
res["corr_dist_time"] = float(corr.loc["Distance_km", "Delivery_Time_min"])
res["corr_dist_cost"] = float(corr.loc["Distance_km", "Shipping_Cost"])
res["corr_load_cost"] = float(corr.loc["load_kg", "Shipping_Cost"])
res["corr_rating_time"] = float(corr.loc["Agent_Rating", "Delivery_Time_min"])
res["late_rate"] = round(float(df["is_late"].mean() * 100), 1)

# >>> charts
# ---- Step 2: Visualizations ----
# 1. Histogram: distribution of delivery time
plt.figure(figsize=(7, 4))
sns.histplot(df["Delivery_Time_min"], bins=30, kde=True, color="#2E75B6")
plt.axvline(60, color="red", linestyle="--", label="Promised time (60 min)")
plt.title("Distribution of Delivery Time"); plt.xlabel("Delivery time (min)"); plt.legend()
plt.tight_layout(); plt.savefig("charts/1_delivery_time_hist.png", dpi=150); plt.close()

# 2. Box plot: delivery time by traffic level
plt.figure(figsize=(7, 4))
sns.boxplot(data=df, x="Traffic", y="Delivery_Time_min", order=["Low", "Medium", "High"], palette="Blues")
plt.title("Delivery Time by Traffic Level"); plt.ylabel("Delivery time (min)")
plt.tight_layout(); plt.savefig("charts/2_traffic_boxplot.png", dpi=150); plt.close()

# 3. Scatter + regression line: distance vs delivery time
plt.figure(figsize=(7, 4))
sns.regplot(data=df.sample(800, random_state=1), x="Distance_km", y="Delivery_Time_min",
            scatter_kws={"alpha": 0.4, "s": 15}, line_kws={"color": "red"})
plt.title("Distance vs Delivery Time"); plt.xlabel("Distance (km)"); plt.ylabel("Delivery time (min)")
plt.tight_layout(); plt.savefig("charts/3_distance_scatter.png", dpi=150); plt.close()

# 4. Line chart: average delivery time by hour of day (finds peak-hour bottlenecks)
hourly = df.groupby("hour")["Delivery_Time_min"].mean()
plt.figure(figsize=(7, 4))
plt.plot(hourly.index, hourly.values, marker="o", color="#1F3864")
plt.title("Average Delivery Time by Hour of Day"); plt.xlabel("Hour"); plt.ylabel("Minutes")
plt.tight_layout(); plt.savefig("charts/4_hourly_trend.png", dpi=150); plt.close()

# 5. Heatmap: correlation matrix
plt.figure(figsize=(6, 4.5))
sns.heatmap(corr, annot=True, cmap="Blues", vmin=-1, vmax=1)
plt.title("Correlation Heatmap"); plt.tight_layout()
plt.savefig("charts/5_correlation_heatmap.png", dpi=150); plt.close()

# 6. Bar chart: late-delivery rate by weather and vehicle
late = df.groupby(["Weather", "Vehicle_Type"])["is_late"].mean().mul(100).reset_index()
plt.figure(figsize=(7, 4))
sns.barplot(data=late, x="Weather", y="is_late", hue="Vehicle_Type", palette="Blues")
plt.title("Late-Delivery Rate by Weather and Vehicle"); plt.ylabel("Late deliveries (%)")
plt.tight_layout(); plt.savefig("charts/6_late_rate_weather.png", dpi=150); plt.close()
# <<< charts

# insights data
res["traffic_mean"] = df.groupby("Traffic")["Delivery_Time_min"].mean().round(1).to_dict()
res["weather_mean"] = df.groupby("Weather")["Delivery_Time_min"].mean().round(1).to_dict()
res["weather_late"] = df.groupby("Weather")["is_late"].mean().mul(100).round(1).to_dict()
res["vehicle_late"] = df.groupby("Vehicle_Type")["is_late"].mean().mul(100).round(1).to_dict()
res["vehicle_cost"] = df.groupby("Vehicle_Type")["Shipping_Cost"].mean().round(1).to_dict()
res["peak_mean"] = df.groupby("is_peak_hour")["Delivery_Time_min"].mean().round(1).to_dict()
res["peak_late"] = df.groupby("is_peak_hour")["is_late"].mean().mul(100).round(1).to_dict()
res["worst_hours"] = hourly.sort_values(ascending=False).head(3).round(1).to_dict()
res["best_hour"] = hourly.sort_values().head(1).round(1).to_dict()
res["high_rain_late"] = round(float(df[(df.Traffic == "High") & (df.Weather == "Rain")]["is_late"].mean() * 100), 1)
res["low_clear_late"] = round(float(df[(df.Traffic == "Low") & (df.Weather == "Clear")]["is_late"].mean() * 100), 1)
json.dump(res, open("week3_results.json", "w"), indent=1, default=str)
print(json.dumps({k: v for k, v in res.items() if k != "describe"}, indent=1, default=str))
