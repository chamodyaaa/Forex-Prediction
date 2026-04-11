import pandas as pd
import seaborn as sns
import streamlit as st
import streamlit.components.v1 as components
import matplotlib.pyplot as plt
from ta.momentum import RSIIndicator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


st.set_page_config(page_title="Forex Prediction Dashboard", layout="wide")


@st.cache_data
def load_and_prepare_data(csv_path: str) -> pd.DataFrame:
    data = pd.read_csv(csv_path)

    data["Datetime"] = pd.to_datetime(data["Date"] + " " + data["Time"])
    data = data.drop(["Date", "Time"], axis=1)
    data = data.sort_values("Datetime").reset_index(drop=True)
    data = data.dropna()

    if "BC" in data.columns:
        data["Close"] = data["BC"]
    elif "Close" not in data.columns:
        raise ValueError("Expected either 'BC' or 'Close' column in CSV.")

    data["Target"] = (data["Close"].shift(-1) > data["Close"]).astype(int)
    data["MA_5"] = data["Close"].rolling(5).mean()
    data["MA_10"] = data["Close"].rolling(10).mean()
    data["MA_20"] = data["Close"].rolling(20).mean()
    data["Volatility"] = data["Close"].rolling(10).std()
    data["RSI"] = RSIIndicator(close=data["Close"], window=14).rsi()
    data["Return"] = data["Close"].pct_change()

    data["Close_lag1"] = data["Close"].shift(1)
    data["Close_lag2"] = data["Close"].shift(2)
    data["Close_lag3"] = data["Close"].shift(3)
    data["RSI_lag1"] = data["RSI"].shift(1)
    data["RSI_lag2"] = data["RSI"].shift(2)
    data["Vol_lag1"] = data["Volatility"].shift(1)
    data["Momentum"] = data["Close"] - data["Close_lag1"]
    data["MA_diff"] = data["MA_5"] - data["MA_20"]

    return data.dropna().reset_index(drop=True)


@st.cache_data
def run_models(data: pd.DataFrame, test_size: float):
    baseline_features = ["MA_5", "MA_10", "MA_20", "Volatility", "RSI"]
    full_features = [
        "MA_5",
        "MA_10",
        "MA_20",
        "Volatility",
        "RSI",
        "Close_lag1",
        "Close_lag2",
        "Close_lag3",
        "RSI_lag1",
        "RSI_lag2",
        "Vol_lag1",
        "Momentum",
        "Return",
        "MA_diff",
    ]

    y = data["Target"]

    # Baseline split
    X_base = data[baseline_features]
    Xb_train, Xb_test, yb_train, yb_test = train_test_split(
        X_base, y, test_size=test_size, shuffle=False
    )
    base_scaler = StandardScaler()
    Xb_train_scaled = base_scaler.fit_transform(Xb_train)
    Xb_test_scaled = base_scaler.transform(Xb_test)

    baseline_model = LogisticRegression(max_iter=1000)
    baseline_model.fit(Xb_train_scaled, yb_train)
    y_pred_baseline = baseline_model.predict(Xb_test_scaled)
    prob_baseline = baseline_model.predict_proba(Xb_test_scaled)[:, 1]

    rf_sample_size = min(20000, len(Xb_train_scaled))
    Xb_train_rf = Xb_train_scaled[-rf_sample_size:]
    yb_train_rf = yb_train.iloc[-rf_sample_size:]

    rf_model = RandomForestClassifier(n_estimators=10, max_depth=8, random_state=42, n_jobs=1)
    rf_model.fit(Xb_train_rf, yb_train_rf)
    y_pred_rf = rf_model.predict(Xb_test_scaled)
    prob_rf = rf_model.predict_proba(Xb_test_scaled)[:, 1]

    # Full model split
    X_full = data[full_features]
    X_train, X_test, y_train, y_test = train_test_split(
        X_full, y, test_size=test_size, shuffle=False
    )

    full_scaler = StandardScaler()
    X_train_scaled = full_scaler.fit_transform(X_train)
    X_test_scaled = full_scaler.transform(X_test)

    gb_model = HistGradientBoostingClassifier(max_iter=100, random_state=42)
    gb_model.fit(X_train_scaled, y_train)

    y_pred_gb = gb_model.predict(X_test_scaled)
    prob_gb = gb_model.predict_proba(X_test_scaled)[:, 1]

    model_results = {
        "Logistic Regression": {
            "accuracy": accuracy_score(yb_test, y_pred_baseline),
            "confusion_matrix": confusion_matrix(yb_test, y_pred_baseline),
            "report": classification_report(yb_test, y_pred_baseline, output_dict=False),
            "test_index": Xb_test.index,
            "probs": prob_baseline,
        },
        "Random Forest": {
            "accuracy": accuracy_score(yb_test, y_pred_rf),
            "confusion_matrix": confusion_matrix(yb_test, y_pred_rf),
            "report": classification_report(yb_test, y_pred_rf, output_dict=False),
            "test_index": Xb_test.index,
            "probs": prob_rf,
        },
        "Gradient Boosting": {
            "accuracy": accuracy_score(y_test, y_pred_gb),
            "confusion_matrix": confusion_matrix(y_test, y_pred_gb),
            "report": classification_report(y_test, y_pred_gb, output_dict=False),
            "test_index": X_test.index,
            "probs": prob_gb,
        },
    }

    results = {
        "baseline_accuracy": model_results["Logistic Regression"]["accuracy"],
        "rf_accuracy": model_results["Random Forest"]["accuracy"],
        "gb_accuracy": model_results["Gradient Boosting"]["accuracy"],
        "class_balance": y.value_counts(normalize=True),
        "full_data": data,
        "model_results": model_results,
    }
    return results


def plot_price_with_ma(data: pd.DataFrame, recent_points: int):
    fig, ax = plt.subplots(figsize=(12, 4))
    sample = data.tail(recent_points)
    ax.plot(sample["Datetime"], sample["Close"], label="Close", linewidth=1.3)
    ax.plot(sample["Datetime"], sample["MA_5"], label="MA_5")
    ax.plot(sample["Datetime"], sample["MA_20"], label="MA_20")
    ax.set_title(f"EUR/USD Price with Moving Averages (Last {recent_points} rows)")
    ax.set_xlabel("Datetime")
    ax.set_ylabel("Price")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_target_distribution(data: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(5, 3.5))
    sns.countplot(x="Target", data=data, ax=ax)
    ax.set_title("Target Distribution")
    ax.set_xlabel("Target (0=Down, 1=Up)")
    ax.set_ylabel("Count")
    fig.tight_layout()
    return fig


def plot_confusion_matrix(cm, model_name: str):
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_title(f"{model_name} Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    fig.tight_layout()
    return fig


def plot_prediction_probabilities(data: pd.DataFrame, test_index, probs, recent_points: int, model_name: str):
    pred_frame = data.loc[test_index, ["Datetime"]].copy()
    pred_frame["Predicted_Prob_Up"] = probs
    sample = pred_frame.tail(recent_points)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(sample["Datetime"], sample["Predicted_Prob_Up"], label="Predicted P(Up)")
    ax.axhline(0.5, linestyle="--", linewidth=1, color="red", label="Threshold 0.5")
    ax.set_title(f"{model_name} Prediction Probability Over Time")
    ax.set_xlabel("Datetime")
    ax.set_ylabel("Probability")
    ax.legend()
    fig.tight_layout()
    return fig


def render_tradingview_widget(symbol: str, interval: str, widget_height: int, theme: str = "light"):
    html = f"""
    <div class="tradingview-widget-container" style="height:{widget_height}px;width:100%;">
        <div id="tradingview_chart" style="height:{widget_height}px;width:100%;"></div>
        <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
        <script type="text/javascript">
            new TradingView.widget({{
                "autosize": true,
                "symbol": "{symbol}",
                "interval": "{interval}",
                "timezone": "Etc/UTC",
                "theme": "{theme}",
                "style": "1",
                "locale": "en",
                "toolbar_bg": "#f1f3f6",
                "enable_publishing": false,
                "hide_top_toolbar": false,
                "hide_legend": false,
                "save_image": false,
                "container_id": "tradingview_chart"
            }});
        </script>
    </div>
    """
    components.html(html, height=widget_height + 25, scrolling=False)


def main():
    st.title("Forex Prediction Dashboard")
    st.caption("Interactive dashboard for model outputs and visual analysis")

    data = load_and_prepare_data("eurusd_hour.csv")
    min_date = data["Datetime"].dt.date.min()
    max_date = data["Datetime"].dt.date.max()

    with st.sidebar:
        st.header("Settings")
        date_range = st.date_input(
            "Date range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
        )
        model_choice = st.selectbox(
            "Model selector",
            options=["Logistic Regression", "Random Forest", "Gradient Boosting"],
            index=2,
        )
        test_size = st.slider("Test size", min_value=0.1, max_value=0.4, value=0.2, step=0.05)
        recent_points = st.slider("Recent rows for charts", min_value=100, max_value=1000, value=400, step=50)
        show_tradingview = st.checkbox("Show TradingView chart", value=True)
        tradingview_symbols = {
            "EUR/USD (OANDA)": "OANDA:EURUSD",
            "GBP/USD (OANDA)": "OANDA:GBPUSD",
            "USD/JPY (OANDA)": "OANDA:USDJPY",
            "Gold (OANDA)": "OANDA:XAUUSD",
        }
        tradingview_symbol = st.selectbox(
            "TradingView symbol",
            options=list(tradingview_symbols.keys()),
            index=0,
        )
        tradingview_interval = st.selectbox(
            "TradingView timeframe",
            options=["15", "60", "240", "D"],
            index=1,
        )
        tradingview_height = st.slider("TradingView height", min_value=600, max_value=1100, value=850, step=50)
        tradingview_symbol = tradingview_symbols[tradingview_symbol]

    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date, end_date = min_date, max_date

    filtered_data = data[
        (data["Datetime"].dt.date >= start_date) & (data["Datetime"].dt.date <= end_date)
    ].copy()

    if len(filtered_data) < 300:
        st.warning("Selected date range is too small. Please choose a wider range.")
        st.stop()

    results = run_models(filtered_data, test_size)
    selected_model = results["model_results"][model_choice]

    c1, c2, c3 = st.columns(3)
    c1.metric("Logistic Accuracy", f"{results['baseline_accuracy']:.4f}")
    c2.metric("RF Accuracy", f"{results['rf_accuracy']:.4f}")
    c3.metric("GB Accuracy", f"{results['gb_accuracy']:.4f}")

    if show_tradingview:
        st.subheader("TradingView Chart")
        render_tradingview_widget(tradingview_symbol, tradingview_interval, tradingview_height)

    st.subheader("Class Balance")
    st.dataframe(results["class_balance"].rename("proportion"))

    left, right = st.columns(2)
    with left:
        st.pyplot(plot_price_with_ma(results["full_data"], recent_points))
        st.pyplot(plot_target_distribution(results["full_data"]))

    with right:
        st.pyplot(plot_confusion_matrix(selected_model["confusion_matrix"], model_choice))
        st.pyplot(
            plot_prediction_probabilities(
                results["full_data"],
                selected_model["test_index"],
                selected_model["probs"],
                recent_points,
                model_choice,
            )
        )

    st.subheader(f"{model_choice} Classification Report")
    st.code(selected_model["report"])


if __name__ == "__main__":
    main()
