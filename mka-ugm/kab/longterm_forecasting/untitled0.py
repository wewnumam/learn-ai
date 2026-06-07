# -*- coding: utf-8 -*-
"""Extended stock analysis and forecasting script for IDX.

pip install yfinance ta tensorflow scikit-learn pandas numpy matplotlib seaborn pmdarima neuralforecast
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import yfinance as yf

from IPython.display import display

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler

try:
    from pmdarima import auto_arima
    PMDARIMA_AVAILABLE = True
except ImportError:
    auto_arima = None  # type: ignore[assignment]
    PMDARIMA_AVAILABLE = False

try:
    from neuralforecast import NeuralForecast
    from neuralforecast.models import NHITS
    NEURALFORECAST_AVAILABLE = True
except ImportError:
    NeuralForecast = None  # type: ignore[assignment]
    NHITS = None  # type: ignore[assignment]
    NEURALFORECAST_AVAILABLE = False

try:
    import ta
    TA_AVAILABLE = True
except ImportError:
    ta = None  # type: ignore[assignment]
    TA_AVAILABLE = False
    logging.warning(
        "The `ta` package is not installed. Technical indicator features will be computed using pandas fallbacks."
    )

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
sns.set(style="whitegrid")

TICKERS: List[str] = ["GOTO.JK", "MTDL.JK", "DCII.JK", "MLPT.JK", "BUKA.JK"]
TARGET_TICKER: str = "GOTO.JK"
LOOK_BACK: int = 60
FORECAST_HORIZON: int = 30
RISK_FREE_RATE: float = 0.05


def download_data(tickers: List[str], period: str = "5y") -> pd.DataFrame:
    """Download historical OHLCV data for given tickers."""
    data = yf.download(tickers, period=period, progress=False)
    if data.empty:
        raise ValueError("Downloaded stock data is empty. Check ticker symbols or connectivity.")
    return data


def compute_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series]:
    """Compute MACD and signal line using exponential moving averages."""
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    macd_signal = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, macd_signal


def compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """Compute a basic RSI series from price data."""
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)


def compute_bollinger_bands(series: pd.Series, window: int = 20, num_std: int = 2) -> Tuple[pd.Series, pd.Series]:
    """Compute lower and upper Bollinger Bands for a price series."""
    ma = series.rolling(window=window).mean()
    std = series.rolling(window=window).std()
    return ma - num_std * std, ma + num_std * std


def compute_support_resistance(series: pd.Series, window: int = 20) -> Tuple[pd.Series, pd.Series]:
    """Compute rolling support and resistance levels as min/max over a window."""
    support = series.rolling(window=window).min()
    resistance = series.rolling(window=window).max()
    return support, resistance


def compute_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical indicators for the target stock dataframe."""
    target_df = df.copy()
    target_df["MA50"] = target_df["Close"].rolling(50).mean()
    target_df["MA200"] = target_df["Close"].rolling(200).mean()
    target_df["EMA12"] = target_df["Close"].ewm(span=12, adjust=False).mean()
    target_df["EMA26"] = target_df["Close"].ewm(span=26, adjust=False).mean()
    target_df["Support"], target_df["Resistance"] = compute_support_resistance(target_df["Close"], window=20)

    if TA_AVAILABLE and ta is not None:
        target_df["MACD"] = ta.trend.macd(target_df["Close"])
        target_df["MACD_Signal"] = ta.trend.macd_signal(target_df["Close"])
        target_df["RSI"] = ta.momentum.rsi(target_df["Close"])
        target_df["BBL"] = ta.volatility.bollinger_lband(target_df["Close"])
        target_df["BBH"] = ta.volatility.bollinger_hband(target_df["Close"])
    else:
        target_df["MACD"], target_df["MACD_Signal"] = compute_macd(target_df["Close"])
        target_df["RSI"] = compute_rsi(target_df["Close"])
        target_df["BBL"], target_df["BBH"] = compute_bollinger_bands(target_df["Close"])

    target_df = target_df.dropna()
    return target_df


def create_lstm_dataset(df: pd.DataFrame, features: List[str], look_back: int) -> Tuple[np.ndarray, np.ndarray]:
    """Create LSTM sequences from raw feature values without scaling."""
    data = df[features].values
    X: List[np.ndarray] = []
    y: List[float] = []
    for i in range(look_back, len(data)):
        X.append(data[i - look_back : i, :])
        y.append(data[i, 0])
    return np.array(X), np.array(y)


def scale_lstm_dataset(
    X_train: np.ndarray,
    X_test: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, MinMaxScaler]:
    """Fit a scaler on training LSTM features and apply it to both train and test."""
    scaler = MinMaxScaler(feature_range=(0, 1))
    flat_train = X_train.reshape(-1, X_train.shape[-1])
    scaler.fit(flat_train)
    X_train_scaled = scaler.transform(flat_train).reshape(X_train.shape)
    X_test_scaled = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
    return X_train_scaled, X_test_scaled, scaler


def build_lstm_model(
    input_shape: Tuple[int, int],
    units1: int = 50,
    units2: int = 50,
    dropout_rate: float = 0.2,
    dense_units: int = 25,
    learning_rate: float = 1e-3,
) -> tf.keras.Model:
    """Create a 2-layer LSTM model with configurable hyperparameters."""
    model = tf.keras.Sequential(
        [
            tf.keras.layers.LSTM(units1, return_sequences=True, input_shape=input_shape),
            tf.keras.layers.Dropout(dropout_rate),
            tf.keras.layers.LSTM(units2, return_sequences=False),
            tf.keras.layers.Dropout(dropout_rate),
            tf.keras.layers.Dense(dense_units),
            tf.keras.layers.Dense(1),
        ]
    )
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss="mean_squared_error")
    return model


def tune_lstm_hyperparameters(
    df_target: pd.DataFrame,
    features: List[str],
    look_back: int,
    train_ratio: float = 0.8,
) -> Dict[str, Any]:
    """Return fixed LSTM model hyperparameters without tuning."""
    X, _ = create_lstm_dataset(df_target, features, look_back)
    if len(X) == 0:
        raise ValueError("Not enough data to create LSTM sequences.")

    fixed_params = {
        "units1": 32,
        "units2": 50,
        "dropout_rate": 0.2457,
        "dense_units": 25,
        "learning_rate": 0.00267,
        "batch_size": 16,
        "epochs": 20,
    }
    logging.info("Using fixed LSTM parameters: %s", fixed_params)
    return {"rmse": None, "params": fixed_params, "history": None}


def tune_nhits_hyperparameters(
    y_train: np.ndarray,
    train_dates: pd.DatetimeIndex,
    forecast_horizon: int,
) -> Dict[str, Any]:
    """Return fixed N-HiTS model hyperparameters without tuning."""
    if not NEURALFORECAST_AVAILABLE or NeuralForecast is None or NHITS is None:
        logging.warning("neuralforecast is not installed. Skipping N-HiTS model.")
        return {"best_params": None, "best_score": None, "results": []}

    fixed_params = {
        "max_steps": 200,
        "learning_rate": 1e-3,
    }
    logging.info("Using fixed N-HiTS parameters: %s", fixed_params)
    return {"best_params": fixed_params, "best_score": None, "results": [{"params": fixed_params, "val_loss": None}]}


def plot_loss_history(history: Dict[str, List[float]], title: str = "Training and Validation Loss") -> None:
    """Plot training and validation loss curves from a history dictionary."""
    if not history:
        return

    plt.figure(figsize=(10, 6))
    plotted = False
    for label, values in history.items():
        if values is None or len(values) == 0:
            continue
        plt.plot(values, label=label)
        plotted = True

    if not plotted:
        plt.close()
        return

    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute regression metrics including directional accuracy."""
    y_true_safe = y_true.copy()
    y_pred_safe = y_pred.copy()
    mae = mean_absolute_error(y_true_safe, y_pred_safe)
    mse = mean_squared_error(y_true_safe, y_pred_safe)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true_safe, y_pred_safe)
    with np.errstate(divide="ignore", invalid="ignore"):
        mape = np.mean(np.abs((y_true_safe - y_pred_safe) / y_true_safe)) * 100
    actual_direction = np.sign(np.diff(y_true_safe))
    predicted_direction = np.sign(np.diff(y_pred_safe))
    dstat = float(np.mean(actual_direction == predicted_direction) * 100) if len(actual_direction) > 0 else np.nan
    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "MAPE": float(mape),
        "R2": float(r2),
        "DSTAT": float(dstat),
    }


def naive_baseline(y_train: np.ndarray, horizon: int) -> np.ndarray:
    """Produce a naive baseline forecast using the last observed training value."""
    if len(y_train) == 0:
        return np.zeros(horizon, dtype=float)
    return np.full(horizon, y_train[-1], dtype=float)


def naive_future_forecast(y_full: np.ndarray, horizon: int) -> np.ndarray:
    """Forecast the next horizon using the last observed value."""
    if len(y_full) == 0:
        return np.zeros(horizon)
    return np.full(horizon, y_full[-1], dtype=float)


def fit_arima(y_train: np.ndarray, n_periods: int) -> Optional[np.ndarray]:
    """Fit an ARIMA model if pmdarima is available."""
    if not PMDARIMA_AVAILABLE or auto_arima is None:
        logging.warning("pmdarima is not installed. Skipping ARIMA model.")
        return None

    try:
        arima_model = auto_arima(
            y_train,
            seasonal=False,
            stepwise=True,
            suppress_warnings=True,
            error_action="ignore",
        )
        return arima_model.predict(n_periods=n_periods)
    except Exception as exc:
        logging.warning("ARIMA fitting failed: %s", exc)
        return None


def fit_nhits(
    y_train: np.ndarray,
    train_dates: pd.DatetimeIndex,
    forecast_horizon: int,
    nhits_config: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[np.ndarray], Optional[Dict[str, List[float]]]]:
    """Fit an N-HiTS model if neuralforecast is available and return forecast plus loss history."""
    if not NEURALFORECAST_AVAILABLE or NeuralForecast is None or NHITS is None:
        logging.warning("neuralforecast is not installed. Skipping N-HiTS model.")
        return None, None

    try:
        df_train = pd.DataFrame(
            {
                "unique_id": ["IDX"] * len(train_dates),
                "ds": train_dates,
                "y": y_train,
            }
        )
        if len(y_train) <= forecast_horizon:
            logging.warning(
                "Not enough training data for N-HiTS: train length %d <= horizon %d. Skipping N-HiTS model.",
                len(y_train),
                forecast_horizon,
            )
            return None, None

        val_size = forecast_horizon
        if len(y_train) - val_size < LOOK_BACK:
            logging.warning(
                "Not enough training history for N-HiTS after setting val_size=%d. Skipping N-HiTS model.",
                val_size,
            )
            return None, None

        try:
            from pytorch_lightning.callbacks import Callback
        except ImportError:
            Callback = None

        class LossHistoryCallback(Callback):
            def __init__(self) -> None:
                super().__init__()
                self.train_loss: List[float] = []
                self.valid_loss: List[float] = []

            def on_train_epoch_end(self, trainer, pl_module) -> None:
                metrics = trainer.callback_metrics
                if "train_loss_epoch" in metrics:
                    self.train_loss.append(float(metrics["train_loss_epoch"]))

            def on_validation_epoch_end(self, trainer, pl_module) -> None:
                metrics = trainer.callback_metrics
                if "valid_loss" in metrics:
                    self.valid_loss.append(float(metrics["valid_loss"]))

        callbacks = [LossHistoryCallback()] if Callback is not None else []

        nhits_kwargs: Dict[str, Any] = {
            "h": forecast_horizon,
            "input_size": LOOK_BACK,
            "batch_size": 32,
            "max_steps": 200,
            "early_stop_patience_steps": 5,
            "val_check_steps": 10,
            "learning_rate": 1e-3,
        }
        if nhits_config is not None:
            nhits_kwargs.update(nhits_config)
        model = NHITS(
            **nhits_kwargs,
            callbacks=callbacks,
            enable_checkpointing=False,
            logger=False,
        )
        nf = NeuralForecast(models=[model], freq="D")
        nf.fit(df=df_train, val_size=val_size)

        history: Optional[Dict[str, List[float]]] = None
        if callbacks:
            loss_callback = callbacks[0]
            history = {
                "N-HiTS Train": loss_callback.train_loss,
                "N-HiTS Val": loss_callback.valid_loss,
            }

        future_dates = pd.date_range(start=train_dates[-1] + pd.Timedelta(days=1), periods=forecast_horizon, freq="D")
        df_future = pd.DataFrame(
            {
                "unique_id": ["IDX"] * len(future_dates),
                "ds": future_dates,
            }
        )
        forecast_df = nf.predict(futr_df=df_future)
        if "NHITS" in forecast_df.columns:
            return forecast_df["NHITS"].values, history
        if forecast_df.shape[1] > 0:
            return forecast_df.iloc[:, -1].values, history

        logging.warning("Unexpected N-HiTS forecast format.")
        return None, history
    except Exception as exc:
        logging.warning("N-HiTS forecasting failed: %s", exc)
        return None, None


def mc_dropout_predictions(model: tf.keras.Model, x: np.ndarray, runs: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    """Return mean and std predictions using MC dropout."""
    x_tensor = tf.convert_to_tensor(x, dtype=tf.float32)
    predictions = [model(x_tensor, training=True).numpy().reshape(-1) for _ in range(runs)]
    pred_array = np.stack(predictions, axis=0)
    return pred_array.mean(axis=0), pred_array.std(axis=0)


def forecast_future(
    model: tf.keras.Model,
    scaler: MinMaxScaler,
    df: pd.DataFrame,
    features: List[str],
    look_back: int,
    horizon: int = 30,
    mc_runs: int = 20,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a future forecast and confidence band from the LSTM model."""
    scaled = scaler.transform(df[features].values)
    window = scaled[-look_back:].copy()
    future_means: List[float] = []
    future_stds: List[float] = []

    for _ in range(horizon):
        mean_pred, std_pred = mc_dropout_predictions(model, np.expand_dims(window, axis=0), runs=mc_runs)
        future_means.append(float(mean_pred[0]))
        future_stds.append(float(std_pred[0]))
        next_row = window[-1].copy()
        next_row[0] = float(mean_pred[0])
        window = np.vstack([window[1:], next_row])

    future_matrix = np.zeros((len(future_means), len(features)))
    future_matrix[:, 0] = np.array(future_means)
    forecast = scaler.inverse_transform(future_matrix)[:, 0]
    confidence = np.array(future_stds) * scaler.scale_[0]
    return forecast, confidence


def plot_correlation_heatmap(data: pd.DataFrame, tickers: List[str]) -> None:
    """Plot daily return correlations across tickers."""
    returns = pd.DataFrame({ticker: data["Close"][ticker].pct_change() for ticker in tickers}).dropna()
    corr = returns.corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True)
    plt.title("Daily Return Correlation Heatmap")
    plt.tight_layout()
    plt.show()

def plot_close_prices(all_stocks_data: pd.DataFrame, tickers: List[str]) -> None:
    """Plot the closing prices for all tickers."""
    plt.figure(figsize=(12, 6))
    for ticker in tickers:
        if ('Close', ticker) in all_stocks_data.columns:
            plt.plot(all_stocks_data['Close'][ticker], label=ticker)
        else:
            logging.warning(f"Close price data not found for {ticker}")
    plt.title('Stock Close Prices (Last 5 Years)')
    plt.xlabel('Date')
    plt.ylabel('Close Price (IDR)')
    plt.grid(True)
    plt.legend()
    plt.show()

def plot_all_time_high(all_stocks_data: pd.DataFrame, tickers: List[str]) -> None:
    """Plot all-time high daily closing price for each stock."""
    plt.figure(figsize=(15, 8))
    for ticker in tickers:
        if ('Close', ticker) in all_stocks_data.columns:
            close_prices = all_stocks_data['Close'][ticker].dropna()
            if not close_prices.empty:
                all_time_high_price = close_prices.max()
                all_time_high_date = close_prices.idxmax()

                plt.plot(close_prices, label=ticker)
                plt.scatter(all_time_high_date, all_time_high_price, color='red', s=50, zorder=5)
                plt.annotate(
                    f'ATH: {all_time_high_price:.2f}',
                    xy=(all_time_high_date, all_time_high_price),
                    xytext=(all_time_high_date, all_time_high_price * 1.1),
                    arrowprops=dict(facecolor='black', shrink=0.05),
                    bbox=dict(boxstyle="round,pad=0.3", fc="yellow", ec="b", lw=1, alpha=0.5),
                    fontsize=10,
                    color='black'
                )
            else:
                logging.warning(f"No valid close price data to find ATH for {ticker}")
        else:
            logging.warning(f"Close price data not found for {ticker}")
    plt.title('All-Time High Daily Closing Price for Stocks (Last 5 Years)')
    plt.xlabel('Date')
    plt.ylabel('Close Price (IDR)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_growth(all_stocks_data: pd.DataFrame, tickers: List[str]) -> None:
    """Plot the percentage growth of each stock over the given period."""
    growth_percentages = {}
    for ticker in tickers:
        if ('Close', ticker) in all_stocks_data.columns:
            close_prices = all_stocks_data['Close'][ticker].dropna()
            if len(close_prices) > 1:
                first_price = close_prices.iloc[0]
                last_price = close_prices.iloc[-1]
                if first_price != 0:
                    growth = ((last_price - first_price) / first_price) * 100
                    growth_percentages[ticker] = growth
                else:
                    growth_percentages[ticker] = 0.0
            else:
                growth_percentages[ticker] = 0.0
        else:
            logging.warning(f"Close price data not found for {ticker}")

    growth_df = pd.DataFrame(list(growth_percentages.items()), columns=['Ticker', 'Growth (%)'])

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Ticker', y='Growth (%)', hue='Ticker', data=growth_df, palette='viridis', legend=False)
    plt.title('Growth in Stock Price Over 5 Years (%)')
    plt.xlabel('Stock Ticker')
    plt.ylabel('Percentage Growth (%)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

def plot_volume(all_stocks_data: pd.DataFrame, tickers: List[str]) -> None:
    """Analyze and plot trading volume for each stock."""
    volume_data = {}
    for ticker in tickers:
        if ('Volume', ticker) in all_stocks_data.columns:
            volumes = all_stocks_data['Volume'][ticker].dropna()
            if not volumes.empty:
                volume_data[ticker] = {
                    'Max Volume': volumes.max(),
                    'Min Volume': volumes.min(),
                    'Average Volume': volumes.mean()
                }
            else:
                logging.warning(f"No valid volume data for {ticker}")
        else:
            logging.warning(f"Volume data not found for {ticker}")

    volume_df = pd.DataFrame.from_dict(volume_data, orient='index')
    display(volume_df)

    for ticker in tickers:
        if ('Volume', ticker) in all_stocks_data.columns:
            volumes = all_stocks_data['Volume'][ticker].dropna()
            if not volumes.empty:
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.plot(volumes, label=f'{ticker} Volume')
                max_volume = volumes.max()
                max_volume_date = volumes.idxmax()
                ax.axvline(x=max_volume_date, ls='--', lw='2.2', color='#0aebff', label=f'Max Volume: {max_volume:,.0f} on {max_volume_date.strftime("%Y-%m-%d")}')

                ax.set_title(f'{ticker} Trade Volume Over 5 Years')
                ax.set_xlabel('Date')
                ax.set_ylabel('Volume')
                ax.grid(True)
                ax.legend()
                plt.tight_layout()
                plt.show()
            else:
                logging.warning(f"Skipping volume plot for {ticker} due to no valid data.")
        else:
            logging.warning(f"Skipping volume plot for {ticker} as volume data column not found.")

def plot_volatility(all_stocks_data: pd.DataFrame, tickers: List[str]) -> None:
    """Calculate and plot 30-day rolling historical volatility."""
    plt.figure(figsize=(15, 8))
    for ticker in tickers:
        if ('Close', ticker) in all_stocks_data.columns:
            close_prices = all_stocks_data['Close'][ticker].dropna()
            if not close_prices.empty:
                daily_returns = close_prices.pct_change().dropna()
                volatility = daily_returns.rolling(window=30).std() * 100
                plt.plot(volatility, label=f'{ticker} 30-Day Volatility (%)')
            else:
                logging.warning(f"No valid close price data to calculate volatility for {ticker}")
        else:
            logging.warning(f"Close price data not found for {ticker}")
    plt.title('30-Day Rolling Historical Volatility (Standard Deviation of Daily Returns)')
    plt.xlabel('Date')
    plt.ylabel('Volatility (%)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_daily_return_stats(all_stocks_data: pd.DataFrame, tickers: List[str]) -> None:
    """Plot daily return percentage for each stock, highlighting max and min returns."""
    for ticker in tickers:
        if ('Close', ticker) in all_stocks_data.columns:
            close_prices = all_stocks_data['Close'][ticker].dropna()
            if not close_prices.empty:
                daily_returns = close_prices.pct_change().dropna() * 100
                if not daily_returns.empty:
                    max_return = daily_returns.max()
                    min_return = daily_returns.min()
                    date_max_return = daily_returns.idxmax()
                    date_min_return = daily_returns.idxmin()

                    fig, ax = plt.subplots(figsize=(12, 6))
                    ax.plot(daily_returns, label=f'{ticker} Daily Returns', alpha=0.7)
                    ax.scatter(date_max_return, max_return, color='green', s=100, zorder=5, label=f'Max Return: {max_return:.2f}% on {date_max_return.strftime("%Y-%m-%d")}')
                    ax.scatter(date_min_return, min_return, color='red', s=100, zorder=5, label=f'Min Return: {min_return:.2f}% on {date_min_return.strftime("%Y-%m-%d")}')

                    ax.annotate(f'{max_return:.2f}%', xy=(date_max_return, max_return), xytext=(date_max_return, max_return + 2),
                                arrowprops=dict(facecolor='black', shrink=0.05), bbox=dict(boxstyle="round,pad=0.3", fc="green", ec="g", lw=1, alpha=0.5))
                    ax.annotate(f'{min_return:.2f}%', xy=(date_min_return, min_return), xytext=(date_min_return, min_return - 2),
                                arrowprops=dict(facecolor='black', shrink=0.05), bbox=dict(boxstyle="round,pad=0.3", fc="red", ec="r", lw=1, alpha=0.5))

                    ax.set_title(f'{ticker} Daily Return Percentage (Last 5 Years)')
                    ax.set_xlabel('Date')
                    ax.set_ylabel('Daily Return (%)')
                    ax.grid(True)
                    ax.legend()
                    plt.tight_layout()
                    plt.show()
                else:
                    logging.warning(f"Not enough data to calculate daily returns for {ticker}")
            else:
                logging.warning(f"No valid close price data for daily returns for {ticker}")
        else:
            logging.warning(f"Close price data not found for {ticker}")

def plot_return_histograms(all_stocks_data: pd.DataFrame, tickers: List[str]) -> None:
    """Create a histogram of daily returns for each stock, with the mean highlighted."""
    for ticker in tickers:
        if ('Close', ticker) in all_stocks_data.columns:
            close_prices = all_stocks_data['Close'][ticker].dropna()
            if not close_prices.empty:
                daily_returns = close_prices.pct_change().dropna() * 100
                if not daily_returns.empty:
                    mean_daily_return = daily_returns.mean()

                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.histplot(daily_returns, bins=50, kde=True, ax=ax, color='skyblue')
                    ax.axvline(mean_daily_return, color='red', linestyle='dashed', linewidth=2, label=f'Mean Return: {mean_daily_return:.2f}%')
                    ax.annotate(f'{mean_daily_return:.2f}%', xy=(mean_daily_return, ax.get_ylim()[1] * 0.9),
                                xytext=(mean_daily_return + (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.05, ax.get_ylim()[1] * 0.8),
                                arrowprops=dict(facecolor='black', shrink=0.05),
                                bbox=dict(boxstyle="round,pad=0.3", fc="yellow", ec="r", lw=1, alpha=0.5),
                                fontsize=10,
                                color='black')

                    ax.set_title(f'Histogram of Daily Returns for {ticker}')
                    ax.set_xlabel('Daily Return (%)')
                    ax.set_ylabel('Frequency')
                    ax.legend()
                    plt.tight_layout()
                    plt.show()
                else:
                    logging.warning(f"Not enough data to create histogram for daily returns for {ticker}")
            else:
                logging.warning(f"No valid close price data for daily returns histogram for {ticker}")
        else:
            logging.warning(f"Close price data not found for {ticker}")


def plot_risk_return(all_stocks_data: pd.DataFrame, tickers: List[str]) -> None:
    """Plot investment risk versus return for each stock using daily returns."""
    stats: List[Dict[str, float]] = []
    for ticker in tickers:
        if ('Close', ticker) not in all_stocks_data.columns:
            logging.warning("Close price data not found for %s", ticker)
            continue

        close_prices = all_stocks_data['Close'][ticker].dropna()
        if close_prices.empty:
            logging.warning("No valid close price data to calculate risk/return for %s", ticker)
            continue

        daily_returns = close_prices.pct_change().dropna()
        if daily_returns.empty:
            logging.warning("Not enough data to calculate daily returns for %s", ticker)
            continue

        stats.append(
            {
                'ticker': ticker,
                'Return (%)': float(daily_returns.mean() * 100),
                'Risk (%)': float(daily_returns.std() * 100),
            }
        )

    if not stats:
        logging.warning("No risk-return statistics available for the provided tickers.")
        return

    risk_return_df = pd.DataFrame(stats).set_index('ticker')

    fig, ax = plt.subplots(figsize=(11, 8))
    sns.scatterplot(
        data=risk_return_df.reset_index(),
        x='Risk (%)',
        y='Return (%)',
        hue='ticker',
        palette='tab10',
        s=140,
        ax=ax,
        legend='brief',
    )

    for ticker, row in risk_return_df.iterrows():
        ax.text(row['Risk (%)'] + 0.01, row['Return (%)'] + 0.01, ticker, fontsize=10)

    ax.set_title('Stock Investment Risk vs Return')
    ax.set_xlabel('Daily Return Volatility (Std Dev %)')
    ax.set_ylabel('Mean Daily Return (%)')
    ax.axhline(0, color='gray', linestyle='--', alpha=0.6)
    ax.axvline(risk_return_df['Risk (%)'].mean(), color='gray', linestyle=':', alpha=0.6)
    ax.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.show()

    try:
        display(risk_return_df.style.format('{:.4f}'))
    except Exception:
        logging.info('Risk-return summary:\n%s', risk_return_df)


def plot_target_signals(df_target: pd.DataFrame, ticker: str) -> None:
    """Plot the target stock price with moving averages, support/resistance, and RSI signals."""
    crosses = pd.DataFrame(
        {
            "golden": (df_target["MA50"] > df_target["MA200"]) & (df_target["MA50"].shift(1) <= df_target["MA200"].shift(1)),
            "death": (df_target["MA50"] < df_target["MA200"]) & (df_target["MA50"].shift(1) >= df_target["MA200"].shift(1)),
        },
        index=df_target.index,
    )

    fig, (ax_price, ax_rsi) = plt.subplots(2, 1, figsize=(16, 10), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    ax_price.plot(df_target.index, df_target["Close"], label="Close", color="#0b3d91")
    ax_price.plot(df_target.index, df_target["MA50"], label="MA50", color="#1f77b4", alpha=0.8)
    ax_price.plot(df_target.index, df_target["MA200"], label="MA200", color="#ff7f0e", alpha=0.8)
    ax_price.plot(df_target.index, df_target["Support"], label="Support", color="#2ca02c", linestyle="--", alpha=0.7)
    ax_price.plot(df_target.index, df_target["Resistance"], label="Resistance", color="#d62728", linestyle="--", alpha=0.7)

    ax_price.scatter(
        crosses.index[crosses["golden"]],
        df_target.loc[crosses["golden"], "Close"],
        marker="^",
        color="green",
        s=100,
        label="Golden Cross",
        zorder=5,
    )
    ax_price.scatter(
        crosses.index[crosses["death"]],
        df_target.loc[crosses["death"], "Close"],
        marker="v",
        color="red",
        s=100,
        label="Death Cross",
        zorder=5,
    )

    ax_price.set_title(f"{ticker} Price with Signals")
    ax_price.set_ylabel("Price (IDR)")
    ax_price.legend()
    ax_price.grid(True, alpha=0.4)

    ax_rsi.plot(df_target.index, df_target["RSI"], label="RSI", color="#9467bd")
    ax_rsi.axhline(70, color="red", linestyle="--", alpha=0.7)
    ax_rsi.axhline(30, color="green", linestyle="--", alpha=0.7)
    ax_rsi.scatter(
        df_target.index[df_target["RSI"] >= 70],
        df_target.loc[df_target["RSI"] >= 70, "RSI"],
        color="red",
        s=30,
        label="Overbought",
    )
    ax_rsi.scatter(
        df_target.index[df_target["RSI"] <= 30],
        df_target.loc[df_target["RSI"] <= 30, "RSI"],
        color="green",
        s=30,
        label="Oversold",
    )
    ax_rsi.set_ylabel("RSI")
    ax_rsi.set_xlabel("Date")
    ax_rsi.legend(loc="upper left")
    ax_rsi.grid(True, alpha=0.4)

    plt.tight_layout()
    plt.show()


def compare_models(
    df_target: pd.DataFrame,
    look_back: int = LOOK_BACK,
    train_ratio: float = 0.8,
    lstm_config: Optional[Dict[str, Any]] = None,
    nhits_config: Optional[Dict[str, Any]] = None,
    ticker: str = TARGET_TICKER,
    horizon: int = FORECAST_HORIZON,
) -> Any:
    """Train and compare multiple forecasting models on the target stock and forecast the next horizon."""
    features = ["Close", "MA50", "MA200", "EMA12", "EMA26", "MACD", "MACD_Signal", "RSI", "BBL", "BBH"]
    X, y = create_lstm_dataset(df_target, features, look_back)
    if len(X) == 0:
        raise ValueError("Not enough data to create LSTM sequences. Increase history length or reduce look_back.")

    n_train = int(len(X) * train_ratio)
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]
    X_train_scaled, X_test_scaled, scaler = scale_lstm_dataset(X_train, X_test)

    close_scaler = MinMaxScaler(feature_range=(0, 1))
    close_scaler.fit(y_train.reshape(-1, 1))
    y_train_scaled = close_scaler.transform(y_train.reshape(-1, 1)).reshape(-1)

    dates = df_target.index[look_back:]
    test_dates = dates[n_train:]

    model_predictions: Dict[str, np.ndarray] = {}
    metrics_table: Dict[str, Dict[str, float]] = {}

    model_predictions["Naive"] = naive_baseline(y_train, len(y_test))
    metrics_table["Naive"] = compute_metrics(y_test, model_predictions["Naive"])

    arima_preds = fit_arima(y_train, len(y_test))
    if arima_preds is not None:
        model_predictions["ARIMA"] = arima_preds
        metrics_table["ARIMA"] = compute_metrics(y_test, arima_preds)

    best_lstm = lstm_config or {
        "units1": 50,
        "units2": 50,
        "dropout_rate": 0.2,
        "dense_units": 25,
        "learning_rate": 1e-3,
    }
    lstm_model = build_lstm_model(
        (X_train_scaled.shape[1], X_train_scaled.shape[2]),
        units1=best_lstm["units1"],
        units2=best_lstm["units2"],
        dropout_rate=best_lstm["dropout_rate"],
        dense_units=best_lstm["dense_units"],
        learning_rate=best_lstm["learning_rate"],
    )
    lstm_history_obj = lstm_model.fit(
        X_train_scaled,
        y_train_scaled,
        epochs=25,
        batch_size=32,
        verbose=0,
        validation_split=0.1,
    )
    lstm_preds = lstm_model.predict(X_test_scaled, verbose=0).reshape(-1)
    lstm_preds_actual = close_scaler.inverse_transform(lstm_preds.reshape(-1, 1)).reshape(-1)

    lstm_history = {
        "LSTM Train": lstm_history_obj.history.get("loss", []),
        "LSTM Val": lstm_history_obj.history.get("val_loss", []),
    }
    plot_loss_history(lstm_history, title="LSTM Training and Validation Loss")

    model_predictions["LSTM"] = lstm_preds_actual
    metrics_table["LSTM"] = compute_metrics(y_test, lstm_preds_actual)

    if arima_preds is not None:
        arima_preds_actual = arima_preds
        model_predictions["ARIMA"] = arima_preds_actual
        metrics_table["ARIMA"] = compute_metrics(y_test, arima_preds_actual)

    nhits_preds, nhits_history = fit_nhits(y_train, dates[:n_train], len(y_test), nhits_config=nhits_config)
    if nhits_preds is not None:
        model_predictions["N-HiTS"] = nhits_preds
        metrics_table["N-HiTS"] = compute_metrics(y_test, nhits_preds)
    if nhits_history is not None:
        plot_loss_history(nhits_history, title="N-HiTS Training and Validation Loss")

    # Forecast the next horizon for every model after the last known date.
    future_forecasts: Dict[str, np.ndarray] = {}
    y_full = y
    future_forecasts["Naive"] = naive_future_forecast(y_full, horizon)
    future_arima = fit_arima(y_full, horizon)
    if future_arima is not None:
        future_forecasts["ARIMA"] = future_arima
    future_lstm, future_confidence = forecast_future(
        lstm_model,
        scaler,
        df_target,
        features,
        look_back,
        horizon=horizon,
    )
    future_forecasts["LSTM"] = future_lstm
    future_nhits, _ = fit_nhits(y_full, dates, horizon, nhits_config=nhits_config)
    if future_nhits is not None:
        future_forecasts["N-HiTS"] = future_nhits

    params_table: Dict[str, str] = {
        "Naive": "naive baseline",
        "LSTM": str(best_lstm),
        "N-HiTS": str(nhits_config or {"max_steps": 100, "learning_rate": 1e-3}),
    }
    if arima_preds is not None:
        params_table["ARIMA"] = "auto_arima"

    metric_names = ["MAE", "RMSE", "MAPE", "DSTAT"]
    winner: Dict[str, str] = {}
    for metric in metric_names:
        values = {model: metrics_table[model][metric] for model in metrics_table}
        if metric == "DSTAT":
            best_model = max(values, key=values.get)
        else:
            best_model = min(values, key=values.get)
        winner[metric] = best_model

    metrics_df = pd.DataFrame(metrics_table).T
    metrics_df["Model Params"] = metrics_df.index.map(params_table).fillna("")
    for metric in metric_names:
        metrics_df[f"Winner_{metric}"] = ""
    for metric in metric_names:
        metrics_df.loc[winner[metric], f"Winner_{metric}"] = "<<"

    logging.info("Model comparison metrics:")
    display(metrics_df[["Model Params", *metric_names, *[f"Winner_{m}" for m in metric_names]]].round(4))

    plt.figure(figsize=(14, 7))
    for model_name, preds in model_predictions.items():
        plt.plot(test_dates, preds, label=model_name)
    plt.plot(test_dates, y_test, label="Actual", color="black", linewidth=2, linestyle="--")
    plt.title(f"Model Comparison on {ticker} Test Set")
    plt.xlabel("Date")
    plt.ylabel("Close Price (IDR)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return (
        lstm_model,
        scaler,
        features,
        test_dates,
        y_test,
        lstm_preds_actual,
        lstm_history,
        nhits_history,
        future_forecasts,
        future_confidence,
    )


def compute_portfolio_metrics(data: pd.DataFrame, tickers: List[str], weights: Optional[List[float]] = None) -> pd.DataFrame:
    """Compute portfolio metrics for each ticker and an equal-weight portfolio."""
    close_prices = pd.DataFrame({ticker: data["Close"][ticker] for ticker in tickers})
    daily_returns = close_prices.pct_change().dropna()
    if weights is None:
        weights = [1.0 / len(tickers)] * len(tickers)
    weights_arr = np.array(weights, dtype=float)
    portfolio_returns = daily_returns.dot(weights_arr)

    summary: Dict[str, Dict[str, float]] = {}
    annual_factor = 252.0
    for ticker in tickers:
        mean_ret = daily_returns[ticker].mean()
        vol = daily_returns[ticker].std()
        cumulative = (1 + daily_returns[ticker]).cumprod()
        drawdown = cumulative / cumulative.cummax() - 1
        summary[ticker] = {
            "Annualised Return": float((1 + mean_ret) ** annual_factor - 1),
            "Annualised Volatility": float(vol * np.sqrt(annual_factor)),
            "Sharpe Ratio": float(((1 + mean_ret) ** annual_factor - 1 - RISK_FREE_RATE) / (vol * np.sqrt(annual_factor)) if vol > 0 else np.nan),
            "Max Drawdown": float(drawdown.min()),
        }

    cum_portfolio = (1 + portfolio_returns).cumprod()
    port_drawdown = cum_portfolio / cum_portfolio.cummax() - 1
    summary["Portfolio"] = {
        "Annualised Return": float((1 + portfolio_returns.mean()) ** annual_factor - 1),
        "Annualised Volatility": float(portfolio_returns.std() * np.sqrt(annual_factor)),
        "Sharpe Ratio": float(((1 + portfolio_returns.mean()) ** annual_factor - 1 - RISK_FREE_RATE) / (portfolio_returns.std() * np.sqrt(annual_factor)) if portfolio_returns.std() > 0 else np.nan),
        "Max Drawdown": float(port_drawdown.min()),
    }

    summary_df = pd.DataFrame(summary).T
    print("\nPortfolio metrics:")
    display(summary_df.round(4))
    return summary_df


def plot_forecast_chart(
    df_target: pd.DataFrame,
    test_dates: pd.DatetimeIndex,
    y_test: np.ndarray,
    lstm_preds: np.ndarray,
    future_forecasts: Dict[str, np.ndarray],
    confidence: Optional[np.ndarray],
    ticker: str,
    horizon: int = FORECAST_HORIZON,
) -> None:
    """Plot historical prices, test predictions, and future forecast with confidence band."""
    historical_dates = df_target.index[-90:]
    historical_prices = df_target["Close"].iloc[-90:]
    forecast_dates = pd.date_range(start=df_target.index[-1] + pd.Timedelta(days=1), periods=horizon, freq="D")

    plt.figure(figsize=(16, 8))
    plt.plot(historical_dates, historical_prices, label="Historical Close", color="#0b3d91")
    plt.plot(test_dates, y_test, label="Test Actual", color="#2ca02c", linestyle="--")
    plt.plot(test_dates, lstm_preds, label="LSTM Test Predictions", color="#ff7f0e")
    for model_name, forecast_vals in future_forecasts.items():
        plt.plot(forecast_dates, forecast_vals, label=f"{model_name} {horizon}-Day Forecast")

    if confidence is not None and "LSTM" in future_forecasts:
        plt.fill_between(
            forecast_dates,
            future_forecasts["LSTM"] - confidence,
            future_forecasts["LSTM"] + confidence,
            color="#9467bd",
            alpha=0.2,
            label="LSTM ±1 Std Confidence",
        )
    plt.title(f"{ticker} Forecast with Confidence Interval")
    plt.xlabel("Date")
    plt.ylabel("Price (IDR)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def main() -> None:
    all_stocks_data = download_data(TICKERS, period="5y")
    display(all_stocks_data.head())

    plot_close_prices(all_stocks_data, TICKERS)
    plot_all_time_high(all_stocks_data, TICKERS)
    plot_growth(all_stocks_data, TICKERS)
    plot_volume(all_stocks_data, TICKERS)
    plot_volatility(all_stocks_data, TICKERS)
    plot_daily_return_stats(all_stocks_data, TICKERS)
    plot_return_histograms(all_stocks_data, TICKERS)
    plot_risk_return(all_stocks_data, TICKERS)
    plot_correlation_heatmap(all_stocks_data, TICKERS)

    features = ["Close", "MA50", "MA200", "EMA12", "EMA26", "MACD", "MACD_Signal", "RSI", "BBL", "BBH"]

    summary_results: List[Dict[str, Any]] = []
    for ticker in TICKERS:
        if ("Close", ticker) not in all_stocks_data.columns:
            logging.warning("Close price data not found for %s, skipping.", ticker)
            continue

        logging.info("\n===== Training models for %s =====", ticker)
        target_df = pd.DataFrame(all_stocks_data["Close"][ticker]).dropna()
        target_df.columns = ["Close"]
        target_df = compute_technical_indicators(target_df)
        plot_target_signals(target_df, ticker)

        best_lstm = tune_lstm_hyperparameters(target_df, features, LOOK_BACK)
        logging.info("Using best LSTM hyperparameters for %s: %s", ticker, best_lstm["params"])

        X, y = create_lstm_dataset(target_df, features, LOOK_BACK)
        n_train = int(len(X) * 0.8)
        if n_train < 1:
            logging.warning("Not enough data to train models for %s. Skipping.", ticker)
            continue
        train_dates = target_df.index[LOOK_BACK : LOOK_BACK + n_train]
        forecast_horizon = len(X) - n_train
        best_nhits = tune_nhits_hyperparameters(y[:n_train], train_dates, forecast_horizon)
        logging.info("N-HiTS tuning result for %s: %s", ticker, best_nhits["best_params"])

        lstm_model, scaler, features, test_dates, y_test_actual, lstm_preds_actual, _, _, future_forecasts, future_confidence = compare_models(
            target_df,
            look_back=LOOK_BACK,
            lstm_config=best_lstm["params"],
            nhits_config=best_nhits["best_params"],
            ticker=ticker,
            horizon=FORECAST_HORIZON,
        )
        plot_forecast_chart(
            target_df,
            test_dates,
            y_test_actual,
            lstm_preds_actual,
            future_forecasts,
            future_confidence,
            ticker,
            horizon=FORECAST_HORIZON,
        )

        summary_results.append({
            "ticker": ticker,
            "best_lstm": best_lstm["params"],
            "best_nhits": best_nhits["best_params"],
        })

    compute_portfolio_metrics(all_stocks_data, TICKERS)


if __name__ == "__main__":
    main()
