# Logistics Data Analyst Internship (YuvaIntern)

**Author:** Vaibhav Nandlal Gangurde
**Duration:** 25 Sep 2026 - 23 Oct 2026

Project: **Last-Mile Delivery Optimization for an E-commerce Company** (hypothetical company QuickKart, Pune).
The project uses a simulated dataset with the same structure as the public Amazon Delivery Dataset (Kaggle).

## Contents
| Week | Topic | Report | Code |
|------|-------|--------|------|
| 1 | Strategic Planning and Data Exploration | `Week1_Logistics_Strategic_Planning_Report.docx` | `week1_kpi_and_model.py` |
| 2 | Data Collection, Cleaning and Preprocessing | `Week2_Data_Cleaning_Preprocessing_Report.docx` | `week2_data_cleaning.py` |
| 3 | Advanced Data Analysis and Visualization | `Week3_Data_Analysis_Visualization_Report.docx` | `week3_eda_visualization.py` |
| 4 | Predictive Modeling and Optimization | `Week4_Predictive_Modeling_Optimization_Report.docx` | `week4_prediction_optimization.py` |

## How to run (in order)
```
pip install pandas numpy scikit-learn matplotlib seaborn
python week2_data_cleaning.py          # creates logistics_raw.csv, logistics_clean.csv
python week3_eda_visualization.py      # creates charts/ and week3_results.json
python week4_prediction_optimization.py
```

## Key results
- Cleaning: 3,080 raw rows -> 3,000 clean rows, 0 missing values
- EDA: distance is the main driver of delivery time (r = 0.85) and cost (r = 0.97); rush hours and rain cause most delays
- Model: tuned Gradient Boosting, MAE about 5 minutes, R2 about 0.89
- Optimization: zone clustering + nearest-neighbour routing, late-risk flagging

## Tools
Python, pandas, NumPy, scikit-learn, matplotlib, seaborn.
