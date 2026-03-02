# Forex Prediction Project

This project predicts currency exchange trends using historical Forex data.

## Features

- Moving averages, RSI, volatility indicators
- Predicts if next day is profitable (classification)
- Uses Logistic Regression (beginner-friendly)
- Time series analysis and visualization

## Visual Data Representation

Running `python "app. py"` now generates charts in the `plots/` folder:

- `price_with_moving_averages.png`
- `target_distribution.png`
- `confusion_matrix.png`
- `prediction_probability_over_time.png`

## Dashboard Representation

An interactive dashboard is available in `dashboard.py`.

Run:
`streamlit run dashboard.py`

The dashboard includes:

- Logistic, Random Forest, and Gradient Boosting accuracy cards
- Sidebar date-range filter
- Sidebar model selector (Logistic / RF / GB)
- Class balance table
- Price + moving-average chart
- Target distribution chart
- Confusion matrix heatmap
- Prediction probability-over-time chart
- Full classification report
