"""
Week 4 - Predictive Modeling and Optimization in Logistics Systems
Author: Vaibhav Nandlal Gangurde

Predicts delivery time (minutes) and uses the results to optimize routes and
vehicle allocation.
Input : logistics_clean.csv (from week2_data_cleaning.py)
Run   : pip install pandas numpy scikit-learn matplotlib && python week4_prediction_optimization.py
"""
import json
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.tree import DecisionTreeRegressor

df = pd.read_csv("logistics_clean.csv", parse_dates=["Order_Date"])
os.makedirs("charts", exist_ok=True)
res = {}

# >>> prep
# ---- Step 1: Data preparation ----
features = ["Distance_km", "Traffic", "Weather", "Vehicle_Type", "hour", "is_peak_hour", "load_kg"]
X = pd.get_dummies(df[features], drop_first=True)
y = df["Delivery_Time_min"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# <<< prep
res["train_size"], res["test_size"], res["n_features"] = len(X_train), len(X_test), X.shape[1]

# >>> models
# ---- Step 2: Train and compare models ----
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(max_depth=6, random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
}
rows = []
for name, m in models.items():
    m.fit(X_train, y_train)
    pred = m.predict(X_test)
    cv = cross_val_score(m, X, y, cv=5, scoring="r2").mean()          # 5-fold cross-validation
    rows.append({"Model": name,
                 "MAE": mean_absolute_error(y_test, pred),
                 "RMSE": np.sqrt(mean_squared_error(y_test, pred)),
                 "R2": r2_score(y_test, pred),
                 "CV_R2": cv})
results = pd.DataFrame(rows).round(3)
print(results)
# <<< models
res["model_table"] = results.to_dict("records")

# >>> tuning
# ---- Step 3: Hyperparameter tuning ----
grid = GridSearchCV(
    GradientBoostingRegressor(random_state=42),
    param_grid={"n_estimators": [100, 200], "max_depth": [2, 3, 4], "learning_rate": [0.05, 0.1]},
    cv=5, scoring="neg_mean_absolute_error", n_jobs=-1)
grid.fit(X_train, y_train)
best = grid.best_estimator_
pred = best.predict(X_test)
print("Best params:", grid.best_params_)
print("Tuned MAE: %.2f | RMSE: %.2f | R2: %.3f" % (
    mean_absolute_error(y_test, pred), np.sqrt(mean_squared_error(y_test, pred)), r2_score(y_test, pred)))
# <<< tuning
res["best_params"] = grid.best_params_
res["tuned"] = {"MAE": round(mean_absolute_error(y_test, pred), 2),
                "RMSE": round(float(np.sqrt(mean_squared_error(y_test, pred))), 2),
                "R2": round(r2_score(y_test, pred), 3)}

# charts: feature importance and actual vs predicted
imp = pd.Series(best.feature_importances_, index=X.columns).sort_values().tail(8)
plt.figure(figsize=(7, 4)); imp.plot(kind="barh", color="#2E75B6")
plt.title("Feature Importance (Tuned Gradient Boosting)"); plt.tight_layout()
plt.savefig("charts/7_feature_importance.png", dpi=150); plt.close()
plt.figure(figsize=(5.5, 5)); plt.scatter(y_test, pred, s=10, alpha=0.4)
lim = [y_test.min(), y_test.max()]; plt.plot(lim, lim, "r--")
plt.xlabel("Actual (min)"); plt.ylabel("Predicted (min)"); plt.title("Actual vs Predicted Delivery Time")
plt.tight_layout(); plt.savefig("charts/8_actual_vs_predicted.png", dpi=150); plt.close()
res["top_features"] = imp.sort_values(ascending=False).round(3).head(4).to_dict()

# >>> optimize
# ---- Step 4: Optimization using model insights ----
# 4a. Late-risk flagging: predicted time above promised 60 min
test = df.loc[X_test.index].copy()
test["pred_time"] = pred
test["risk_late"] = test["pred_time"] > 60
res["flagged_pct"] = round(float(test["risk_late"].mean() * 100), 1)
res["flag_precision"] = round(float((test.loc[test.risk_late, "Delivery_Time_min"] > 60).mean() * 100), 1)
res["flag_recall"] = round(float(test.loc[test.Delivery_Time_min > 60, "risk_late"].mean() * 100), 1)

# 4b. Zone-based vehicle allocation: cluster drop points into 8 zones (one rider per zone)
coords = df[["drop_latitude", "drop_longitude"]].values
df["zone"] = KMeans(n_clusters=8, random_state=42, n_init=10).fit_predict(coords)

def route_length(points):                       # total distance of visiting points in order
    return np.sqrt((np.diff(points, axis=0) ** 2).sum(axis=1)).sum() * 111   # degrees -> km

def nearest_neighbour(points):                  # greedy route: always go to the closest next stop
    left, order = list(range(1, len(points))), [0]
    while left:
        nxt = min(left, key=lambda i: np.linalg.norm(points[order[-1]] - points[i]))
        order.append(nxt); left.remove(nxt)
    return points[order]

before = after = 0
for z in range(8):
    pts = df[df.zone == z].head(40)[["drop_latitude", "drop_longitude"]].values   # 40 stops per route
    before += route_length(pts)                  # current: stops visited in order received
    after += route_length(nearest_neighbour(pts))
print("Route distance before: %.0f km | after: %.0f km | saving: %.1f%%" % (
    before, after, (1 - after / before) * 100))
# <<< optimize
res["route_before"], res["route_after"] = round(before), round(after)
res["route_saving"] = round((1 - after / before) * 100, 1)

# cost impact: peak-hour deliveries are ~6 min slower; shifting 20% of non-urgent peak orders
peak_orders = int(df.is_peak_hour.sum())
res["peak_orders"] = peak_orders
res["peak_extra_min"] = round(float(df[df.is_peak_hour == 1].Delivery_Time_min.mean() - df[df.is_peak_hour == 0].Delivery_Time_min.mean()), 1)
json.dump(res, open("week4_results.json", "w"), indent=1, default=str)
print(json.dumps(res, indent=1, default=str))
