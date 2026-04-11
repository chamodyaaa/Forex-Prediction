# Forex Prediction Project

A Python-based Forex analysis project that uses technical indicators and machine learning to predict the next price direction of EUR/USD. The repository includes both a script that generates static charts and an interactive Streamlit dashboard with a TradingView chart.

## Project Overview

The project works with historical hourly Forex data from `eurusd_hour.csv` and builds features such as:

- Moving averages
- RSI
- Volatility
- Momentum
- Lagged price and indicator values

It then trains multiple models to estimate whether the next candle will move up or down.

## Features

- Moving averages, RSI, volatility indicators
- Predicts if next day is profitable 
- Uses Logistic Regression 
- Time series analysis and visualization

- `Date`
- `Time`
- `BC` or `Close`

## Run the Script Version

Run the standalone script to train the models and save plots into the `plots/` folder:

```bash
python "app. py"
```

Generated images include:

- `price_with_moving_averages.png`
- `target_distribution.png`
- `confusion_matrix.png`
- `prediction_probability_over_time.png`

## Run the Dashboard

Launch the interactive dashboard with:

```bash
streamlit run dashboard.py
```

### Dashboard Controls

- Date range filter
- Model selector
- Test-size slider
- Recent rows slider for charts
- TradingView toggle
- TradingView symbol selector
- TradingView timeframe selector
- TradingView height slider

### Dashboard Outputs

- Accuracy cards for Logistic Regression, Random Forest, and Gradient Boosting
- Class balance table
- Price and moving-average chart
- Target distribution chart
- Confusion matrix heatmap
- Prediction probability chart
- Full classification report
- TradingView market chart for visual context

## Notes

- The dashboard predicts next-step direction, not the exact future price.
- The TradingView chart is for live market context and does not use the model itself.
- This project is for analysis and learning, not financial advice.

## Suggested Next Improvements

- Add more symbols beyond EUR/USD
- Add a real-time data feed
- Compare additional models and tune hyperparameters
- Add backtesting metrics such as precision on bullish signals
