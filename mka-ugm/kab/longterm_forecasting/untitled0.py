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
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    optuna = None  # type: ignore[assignment]
    OPTUNA_AVAILABLE = False
    logging.warning(
        "The `optuna` package is not installed. Hyperparameter tuning will fall back to default parameters."
    )

try:
    from scipy.optimize import minimize
    SCIPY_OPTIMIZE_AVAILABLE = True
except ImportError:
    minimize = None  # type: ignore[assignment]
    SCIPY_OPTIMIZE_AVAILABLE = False
    logging.warning(
        "The `scipy` package is not installed. Markowitz optimization will fall back to equal weights."
    )

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

# Configure pandas display options to show full table without cutting/truncating cell content
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.max_colwidth", None)
pd.set_option("display.width", 1000)

TICKERS: List[str] = ['AAPL',  # Apple Inc.
                      'AXP',   # American Express Co.
                      'KO',    # Coca-Cola Co.
                      'BAC',   # Bank of America Corp.
                      'CVX',   # Chevron Corp.
                      'MCO',   # Moody's Corporation
                      'OXY',   # Occidental Petroleum Corp.
                      'CB',    # Chubb Ltd.
                      'KHC',   # Kraft Heinz Co.
                      'GOOGL'  # Alphabet Inc. Class A
                      ]
TARGET_TICKER: str = "AAPL"
LOOK_BACK: int = 60
FORECAST_HORIZON: int = 120  # 6 months * 20 trading days
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
    """Dynamically tune LSTM model hyperparameters using Optuna to minimize validation RMSE."""
    X, y = create_lstm_dataset(df_target, features, look_back)
    if len(X) == 0:
        raise ValueError("Not enough data to create LSTM sequences.")

    n_train = int(len(X) * train_ratio)
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]

    X_train_scaled, X_test_scaled, scaler = scale_lstm_dataset(X_train, X_test)
    close_scaler = MinMaxScaler(feature_range=(0, 1))
    close_scaler.fit(y_train.reshape(-1, 1))
    y_train_scaled = close_scaler.transform(y_train.reshape(-1, 1)).reshape(-1)

    def objective(trial):
        units1 = trial.suggest_int("units1", 32, 128)
        units2 = trial.suggest_int("units2", 32, 128)
        dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.3)
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])

        model = build_lstm_model(
            (X_train_scaled.shape[1], X_train_scaled.shape[2]),
            units1=units1,
            units2=units2,
            dropout_rate=dropout_rate,
            dense_units=25,
            learning_rate=learning_rate,
        )
        model.fit(
            X_train_scaled,
            y_train_scaled,
            epochs=20,
            batch_size=batch_size,
            verbose=0,
            validation_split=0.1,
        )
        preds_scaled = model.predict(X_test_scaled, verbose=0).reshape(-1)
        preds = close_scaler.inverse_transform(preds_scaled.reshape(-1, 1)).reshape(-1)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        return rmse

    if OPTUNA_AVAILABLE and optuna is not None:
        logging.info("Starting Optuna hyperparameter tuning for LSTM...")
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=10)
        best_params = study.best_params
        best_params["dense_units"] = 25
        best_params["epochs"] = 20
        logging.info("Best LSTM parameters found: %s with RMSE: %.4f", best_params, study.best_value)
        return {"rmse": study.best_value, "params": best_params, "history": None}
    else:
        fixed_params = {
            "units1": 32,
            "units2": 50,
            "dropout_rate": 0.2,
            "dense_units": 25,
            "learning_rate": 1e-3,
            "batch_size": 32,
            "epochs": 20,
        }
        logging.info("Optuna not available. Using fixed LSTM parameters: %s", fixed_params)
        return {"rmse": None, "params": fixed_params, "history": None}


def tune_nhits_hyperparameters(
    y_train: np.ndarray,
    train_dates: pd.DatetimeIndex,
    forecast_horizon: int,
) -> Dict[str, Any]:
    """Dynamically tune N-HiTS model hyperparameters using Optuna to minimize validation loss."""
    if not NEURALFORECAST_AVAILABLE or NeuralForecast is None or NHITS is None:
        logging.warning("neuralforecast is not installed. Skipping N-HiTS model.")
        return {"best_params": None, "best_score": None, "results": []}

    if len(y_train) <= forecast_horizon:
        logging.warning("Not enough training data for N-HiTS tuning. Skipping.")
        return {"best_params": None, "best_score": None, "results": []}

    def objective(trial):
        max_steps = trial.suggest_int("max_steps", 100, 500)
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)

        try:
            df_train = pd.DataFrame(
                {
                    "unique_id": ["IDX"] * len(train_dates),
                    "ds": train_dates,
                    "y": y_train,
                }
            )

            class ValLossCallback:
                def __init__(self):
                    self.val_loss = float("inf")
                def on_validation_epoch_end(self, trainer, pl_module):
                    metrics = trainer.callback_metrics
                    if "valid_loss" in metrics:
                        self.val_loss = float(metrics["valid_loss"])

            callback = ValLossCallback()

            model = NHITS(
                h=forecast_horizon,
                input_size=LOOK_BACK,
                batch_size=32,
                max_steps=max_steps,
                early_stop_patience_steps=5,
                val_check_steps=10,
                learning_rate=learning_rate,
                callbacks=[callback],
                enable_checkpointing=False,
                logger=False,
            )
            nf = NeuralForecast(models=[model], freq="D")
            nf.fit(df=df_train, val_size=forecast_horizon)

            return callback.val_loss
        except Exception:
            return float("inf")

    if OPTUNA_AVAILABLE and optuna is not None:
        logging.info("Starting Optuna hyperparameter tuning for N-HiTS...")
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=5)
        best_params = study.best_params
        logging.info("Best N-HiTS parameters found: %s", best_params)
        return {"best_params": best_params, "best_score": study.best_value, "results": []}
    else:
        fixed_params = {
            "max_steps": 200,
            "learning_rate": 1e-3,
        }
        logging.info("Optuna not available. Using fixed N-HiTS parameters: %s", fixed_params)
        return {"best_params": fixed_params, "best_score": None, "results": []}


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
    """Plot investment risk versus return for each stock using annualized returns and volatility."""
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

        # Annualized return = daily mean * 252 * 100
        # Annualized risk = daily std * sqrt(252) * 100
        ann_return = float(daily_returns.mean() * 252 * 100)
        ann_risk = float(daily_returns.std() * np.sqrt(252) * 100)
        sharpe = (ann_return / 100 - RISK_FREE_RATE) / (ann_risk / 100) if ann_risk > 0 else 0.0

        stats.append(
            {
                'ticker': ticker,
                'Annualized Return (%)': ann_return,
                'Annualized Risk (%)': ann_risk,
                'Sharpe Ratio': sharpe,
            }
        )

    if not stats:
        logging.warning("No risk-return statistics available for the provided tickers.")
        return

    risk_return_df = pd.DataFrame(stats).set_index('ticker')

    fig, ax = plt.subplots(figsize=(12, 8))
    # Scatter plot color-coded by Sharpe Ratio using the viridis colormap
    scatter = ax.scatter(
        risk_return_df['Annualized Risk (%)'],
        risk_return_df['Annualized Return (%)'],
        c=risk_return_df['Sharpe Ratio'],
        cmap='viridis',
        s=150,
        edgecolors='black',
        alpha=0.9,
        zorder=3,
    )
    
    # Add colorbar for Sharpe Ratio
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Sharpe Ratio', fontsize=12)

    # Label placement directly on top of the scatter points
    for ticker, row in risk_return_df.iterrows():
        x_val = row['Annualized Risk (%)']
        y_val = row['Annualized Return (%)']
        ax.text(
            x_val,
            y_val + 0.35,  # Offset vertically above the point
            ticker,
            fontsize=10,
            fontweight='bold',
            ha='center',
            va='bottom',
            zorder=4,
        )

    ax.set_title('Stock Investment Annualized Risk vs Return (Sharpe Ratio Analysis)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Annualized Volatility / Risk (%)', fontsize=12)
    ax.set_ylabel('Annualized Expected Return (%)', fontsize=12)
    
    # Reference lines
    ax.axhline(RISK_FREE_RATE * 100, color='red', linestyle='--', alpha=0.7, label=f'Risk-Free Rate ({RISK_FREE_RATE*100:.1f}%)')
    ax.axvline(risk_return_df['Annualized Risk (%)'].mean(), color='gray', linestyle=':', alpha=0.6, label='Mean Risk')
    ax.axhline(risk_return_df['Annualized Return (%)'].mean(), color='gray', linestyle=':', alpha=0.6, label='Mean Return')
    
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left')
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

    # Create two subplots: main prediction comparison on top, residuals on bottom
    fig, (ax_main, ax_res) = plt.subplots(2, 1, figsize=(14, 9), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    
    line_styles = {
        "Naive": ":",
        "ARIMA": "-.",
        "LSTM": "-",
        "N-HiTS": "--",
    }
    
    # Plot Actual close price first with a thick, distinct line on main plot
    ax_main.plot(test_dates, y_test, label="Actual Close", color="black", linewidth=2.5, zorder=3)
    
    # Plot model predictions
    for model_name, preds in model_predictions.items():
        style = line_styles.get(model_name, "-")
        ax_main.plot(test_dates, preds, label=f"{model_name} Forecast", linestyle=style, linewidth=1.8)
        
    ax_main.set_title(f"Model Comparison on {ticker} Test Set", fontsize=14, fontweight='bold')
    ax_main.set_ylabel("Close Price (IDR)", fontsize=12)
    ax_main.legend(loc="upper left")
    ax_main.grid(True, alpha=0.4)
    
    # Create evaluation metrics summary text block
    metrics_text = "Model Test Metrics:\n" + "\n".join(
        f"{model}: MAE={metrics['MAE']:.2f}, RMSE={metrics['RMSE']:.2f}, MAPE={metrics['MAPE']:.2f}%"
        for model, metrics in metrics_table.items()
    )
    
    # Add metrics summary box to the main plot
    ax_main.text(
        0.02, 0.05, metrics_text,
        transform=ax_main.transAxes,
        fontsize=9,
        verticalalignment='bottom',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor='gray')
    )
    
    # Plot residuals on the bottom plot
    for model_name, preds in model_predictions.items():
        residuals = y_test - preds
        style = line_styles.get(model_name, "-")
        ax_res.plot(test_dates, residuals, label=f"{model_name} Residual", linestyle=style, alpha=0.8)
        
    ax_res.axhline(0, color="black", linestyle="--", alpha=0.6)
    ax_res.set_ylabel("Residual (Actual - Pred)", fontsize=11)
    ax_res.set_xlabel("Date", fontsize=12)
    ax_res.grid(True, alpha=0.4)
    ax_res.legend(loc="upper left", fontsize=8)
    
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


def plot_jci_usd_context(period: str = "1y") -> None:
    """Download and plot JCI (^JKSE) and USD/IDR (IDR=X) to show market conditions."""
    logging.info("Downloading JCI (^JKSE) and USD/IDR (IDR=X) exchange rate data for context...")
    try:
        jci = yf.download("^JKSE", period=period, progress=False)
        usd_idr = yf.download("IDR=X", period=period, progress=False)
        
        if jci.empty or usd_idr.empty:
            logging.warning("Could not download context data for JCI or USD/IDR.")
            return

        # Align datasets by index
        combined = pd.DataFrame({
            "JCI": jci["Close"],
            "USD_IDR": usd_idr["Close"]
        }).dropna()

        fig, ax1 = plt.subplots(figsize=(14, 6))

        color = '#0b3d91'
        ax1.set_xlabel('Date', fontsize=12)
        ax1.set_ylabel('JCI Index Value', color=color, fontsize=12)
        ax1.plot(combined.index, combined['JCI'], color=color, label='JCI Index (^JKSE)', linewidth=2)
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.grid(True, alpha=0.3)

        ax2 = ax1.twinx()  
        color = '#d62728'
        ax2.set_ylabel('USD/IDR Exchange Rate', color=color, fontsize=12)
        ax2.plot(combined.index, combined['USD_IDR'], color=color, label='USD/IDR (IDR=X)', linewidth=2, linestyle='--')
        ax2.tick_params(axis='y', labelcolor=color)

        plt.title('Indonesian Market Context: JCI Index vs. USD/IDR Exchange Rate', fontsize=14, fontweight='bold')
        fig.tight_layout()  
        
        # Combine legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        plt.show()
    except Exception as e:
        logging.warning("Failed to plot JCI and USD/IDR context: %s", e)


def get_usd_idr_rate() -> float:
    """Fetch the current USD to IDR exchange rate."""
    try:
        idr_data = yf.download("IDR=X", period="1d", progress=False)
        if not idr_data.empty and "Close" in idr_data.columns:
            return float(idr_data["Close"].iloc[-1])
    except Exception as e:
        logging.warning("Failed to fetch USD/IDR rate: %s. Using fallback 16,000.", e)
    return 16000.0


def markowitz_optimization(
    expected_returns: np.ndarray,
    cov_matrix: np.ndarray,
    risk_free_rate: float = 0.05
) -> np.ndarray:
    """
    Perform Markowitz Mean-Variance Optimization to find the Maximum Sharpe Ratio Portfolio.
    """
    if not SCIPY_OPTIMIZE_AVAILABLE or minimize is None:
        logging.warning("scipy.optimize not available. Falling back to equal weights.")
        return np.ones(len(expected_returns)) / len(expected_returns)

    num_assets = len(expected_returns)
    
    def portfolio_performance(weights):
        port_return = np.sum(expected_returns * weights)
        port_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe_ratio = (port_return - risk_free_rate) / port_volatility
        return -sharpe_ratio  # Minimize negative Sharpe ratio

    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0})
    bounds = tuple((0.0, 0.5) for _ in range(num_assets))  # No short selling, max 50% per asset
    initial_guess = np.ones(num_assets) / num_assets

    result = minimize(
        portfolio_performance,
        initial_guess,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    
    if result.success:
        return result.x
    else:
        logging.warning("Markowitz optimization failed: %s. Falling back to equal weights.", result.message)
        return initial_guess


def analyze_and_allocate_portfolio(
    all_stocks_data: pd.DataFrame,
    tickers: List[str],
    future_forecasts_all: Dict[str, np.ndarray],
    investment_idr: float = 1_000_000_000_000.0,
    top_n: int = 5,
    weighting_method: str = "markowitz",  # "markowitz" or "return_proportional"
    model_name: str = "Unknown"
) -> pd.DataFrame:
    """
    Select top N stocks based on projected 6-month return, allocate weights 
    using the specified method (Markowitz Optimization or Return-Proportional), and calculate expected profit.
    """
    usd_idr_rate = get_usd_idr_rate()
    investment_usd = investment_idr / usd_idr_rate
    logging.info("[%s Model] Current USD/IDR rate: %.2f. Investment in USD: $%.2f", model_name, usd_idr_rate, investment_usd)

    projections: List[Dict[str, Any]] = []
    for ticker in tickers:
        if ticker not in future_forecasts_all:
            continue
        
        forecast = future_forecasts_all[ticker]
        if len(forecast) == 0:
            continue
            
        last_price = all_stocks_data["Close"][ticker].dropna().iloc[-1]
        projected_price = forecast[-1]
        projected_return = (projected_price - last_price) / last_price
        
        projections.append({
            "Ticker": ticker,
            "Last_Price": last_price,
            "Projected_Price_6M": projected_price,
            "Projected_Return": projected_return,
        })

    proj_df = pd.DataFrame(projections)
    if proj_df.empty:
        logging.warning("[%s Model] No projections available for portfolio allocation.", model_name)
        return pd.DataFrame()

    # Sort by projected return descending and select top N
    proj_df = proj_df.sort_values(by="Projected_Return", ascending=False).head(top_n)
    selected_tickers = proj_df["Ticker"].tolist()
    
    # Calculate historical covariance matrix (annualized)
    close_prices = pd.DataFrame({ticker: all_stocks_data["Close"][ticker] for ticker in selected_tickers})
    daily_returns = close_prices.pct_change().dropna()
    cov_matrix = daily_returns.cov().values * 252  # Annualized covariance
    
    # Expected returns
    expected_returns = proj_df["Projected_Return"].values
    
    if weighting_method == "return_proportional":
        # Proportional weights: w_i = R_i / sum(R_j)
        # Shift to positive values if any are negative to maintain proportionality
        min_ret = expected_returns.min()
        if min_ret < 0:
            adjusted_returns = expected_returns - min_ret + 1e-5
        else:
            adjusted_returns = expected_returns
            
        optimal_weights = adjusted_returns / adjusted_returns.sum()
        logging.info("[%s Model] Allocating weights proportional to projected returns...", model_name)
        label = f"{model_name} Return-Proportional Portfolio"
    else:
        # Perform Markowitz Optimization
        optimal_weights = markowitz_optimization(expected_returns, cov_matrix, risk_free_rate=RISK_FREE_RATE)
        logging.info("[%s Model] Allocating weights using Markowitz Optimization (Max Sharpe)...", model_name)
        label = f"{model_name} Max Sharpe Ratio Portfolio"
    
    proj_df["Weight"] = optimal_weights
    proj_df["Allocated_USD"] = investment_usd * optimal_weights
    proj_df["Allocated_IDR"] = investment_idr * optimal_weights
    proj_df["Expected_Profit_USD"] = proj_df["Allocated_USD"] * proj_df["Projected_Return"]
    proj_df["Expected_Profit_IDR"] = proj_df["Expected_Profit_USD"] * usd_idr_rate

    print("\n" + "="*80)
    title_suffix = "Return-Proportional Weighting" if weighting_method == "return_proportional" else "Markowitz Max Sharpe Ratio"
    print(f"TOP {top_n} STOCKS FOR 6-MONTH INVESTMENT ({model_name} Model - {title_suffix})")
    print("="*80)
    display(proj_df.round(4))
    
    total_expected_profit_idr = proj_df["Expected_Profit_IDR"].sum()
    print(f"\nTotal Expected Portfolio Profit in 6 Months: IDR {total_expected_profit_idr:,.2f}")
    print(f"Total Expected Portfolio Profit in 6 Months: USD ${proj_df['Expected_Profit_USD'].sum():,.2f}")
    print("="*80 + "\n")
    
    # Plot Efficient Frontier
    plot_efficient_frontier(daily_returns, optimal_weights, selected_tickers, label=label)
    
    return proj_df


def plot_efficient_frontier(
    daily_returns: pd.DataFrame,
    optimal_weights: np.ndarray,
    tickers: List[str],
    num_portfolios: int = 10000,
    label: str = "Max Sharpe Ratio Portfolio"
) -> None:
    """Plot the Efficient Frontier and mark the designated portfolio."""
    num_assets = len(tickers)
    results = np.zeros((3, num_portfolios))
    
    for i in range(num_portfolios):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        
        port_return = np.sum(daily_returns.mean() * weights) * 252
        port_volatility = np.sqrt(np.dot(weights.T, np.dot(daily_returns.cov() * 252, weights)))
        sharpe_ratio = (port_return - RISK_FREE_RATE) / port_volatility
        
        results[0, i] = port_volatility
        results[1, i] = port_return
        results[2, i] = sharpe_ratio
        
    results_frame = pd.DataFrame(results.T, columns=['Volatility', 'Return', 'Sharpe'])
    
    plt.figure(figsize=(10, 6))
    plt.scatter(results_frame['Volatility'], results_frame['Return'], c=results_frame['Sharpe'], cmap='viridis', marker='o', alpha=0.3)
    plt.colorbar(label='Sharpe Ratio')
    
    # Mark optimal portfolio
    opt_return = np.sum(daily_returns.mean() * optimal_weights) * 252
    opt_volatility = np.sqrt(np.dot(optimal_weights.T, np.dot(daily_returns.cov() * 252, optimal_weights)))
    plt.scatter(opt_volatility, opt_return, color='red', s=100, marker='*', label=label)
    
    plt.title(f'Efficient Frontier with {label}')
    plt.xlabel('Expected Volatility (Annualized)')
    plt.ylabel('Expected Return (Annualized)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def compute_portfolio_metrics(data: pd.DataFrame, tickers: List[str], weights: Optional[List[float]] = None) -> pd.DataFrame:
    """Compute portfolio metrics for each ticker, an equal-weight portfolio, and the optimized portfolio."""
    close_prices = pd.DataFrame({ticker: data["Close"][ticker] for ticker in tickers})
    daily_returns = close_prices.pct_change().dropna()
    
    summary: Dict[str, Dict[str, float]] = {}
    annual_factor = 252.0
    
    # Individual ticker metrics
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

    # Equal-weight portfolio metrics
    equal_weights = np.array([1.0 / len(tickers)] * len(tickers), dtype=float)
    equal_portfolio_returns = daily_returns.dot(equal_weights)
    cum_equal = (1 + equal_portfolio_returns).cumprod()
    port_drawdown_equal = cum_equal / cum_equal.cummax() - 1
    
    summary["Equal Weight Portfolio"] = {
        "Annualised Return": float((1 + equal_portfolio_returns.mean()) ** annual_factor - 1),
        "Annualised Volatility": float(equal_portfolio_returns.std() * np.sqrt(annual_factor)),
        "Sharpe Ratio": float(((1 + equal_portfolio_returns.mean()) ** annual_factor - 1 - RISK_FREE_RATE) / (equal_portfolio_returns.std() * np.sqrt(annual_factor)) if equal_portfolio_returns.std() > 0 else np.nan),
        "Max Drawdown": float(port_drawdown_equal.min()),
    }

    # Optimized portfolio metrics (if weights provided)
    if weights is not None:
        weights_arr = np.array(weights, dtype=float)
        opt_portfolio_returns = daily_returns.dot(weights_arr)
        cum_opt = (1 + opt_portfolio_returns).cumprod()
        port_drawdown_opt = cum_opt / cum_opt.cummax() - 1
        
        summary["Optimized Portfolio (Max Sharpe)"] = {
            "Annualised Return": float((1 + opt_portfolio_returns.mean()) ** annual_factor - 1),
            "Annualised Volatility": float(opt_portfolio_returns.std() * np.sqrt(annual_factor)),
            "Sharpe Ratio": float(((1 + opt_portfolio_returns.mean()) ** annual_factor - 1 - RISK_FREE_RATE) / (opt_portfolio_returns.std() * np.sqrt(annual_factor)) if opt_portfolio_returns.std() > 0 else np.nan),
            "Max Drawdown": float(port_drawdown_opt.min()),
        }

    summary_df = pd.DataFrame(summary).T
    print("\nPortfolio metrics comparison:")
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

    # Connect forecast lines smoothly by prepending the last historical close price and date
    last_hist_date = df_target.index[-1]
    last_hist_val = df_target["Close"].iloc[-1]
    forecast_dates_extended = pd.DatetimeIndex([last_hist_date]).append(forecast_dates)

    plt.figure(figsize=(16, 8))
    
    # Plot historical data and test backtesting lines
    plt.plot(historical_dates, historical_prices, label="Historical Close", color="#0b3d91", linewidth=2)
    plt.plot(test_dates, y_test, label="Test Actual", color="#2ca02c", linestyle="--", linewidth=1.5)
    plt.plot(test_dates, lstm_preds, label="LSTM Test Predictions", color="#ff7f0e", linewidth=1.5)
    
    # Plot future forecasts (connected smoothly)
    for model_name, forecast_vals in future_forecasts.items():
        forecast_extended = np.insert(forecast_vals, 0, last_hist_val)
        plt.plot(forecast_dates_extended, forecast_extended, label=f"{model_name} {horizon}-Day Forecast", linewidth=2)

    # Plot nested confidence bands (68% and 95% confidence intervals)
    if confidence is not None and "LSTM" in future_forecasts:
        lstm_forecast_extended = np.insert(future_forecasts["LSTM"], 0, last_hist_val)
        confidence_extended = np.insert(confidence, 0, 0.0) # No uncertainty at starting point
        
        plt.fill_between(
            forecast_dates_extended,
            lstm_forecast_extended - confidence_extended,
            lstm_forecast_extended + confidence_extended,
            color="#9467bd",
            alpha=0.18,
            label="LSTM ±1 Std Dev (68% CI)",
        )
        plt.fill_between(
            forecast_dates_extended,
            lstm_forecast_extended - 2 * confidence_extended,
            lstm_forecast_extended + 2 * confidence_extended,
            color="#9467bd",
            alpha=0.08,
            label="LSTM ±2 Std Dev (95% CI)",
        )
        
    # Shade backtesting and future forecast regions
    if len(test_dates) > 0:
        plt.axvspan(test_dates[0], test_dates[-1], color="gray", alpha=0.08, label="Backtesting (Test) Period")
    if len(forecast_dates_extended) > 0:
        plt.axvspan(forecast_dates_extended[0], forecast_dates_extended[-1], color="blue", alpha=0.04, label="Future Forecast Period")

    plt.title(f"{ticker} Price Forecast with Confidence Interval Bands", fontsize=14, fontweight='bold')
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Price (IDR)", fontsize=12)
    plt.legend(loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def main() -> None:
    # Plot JCI vs USD/IDR first for market context
    plot_jci_usd_context(period="1y")

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
    # To store forecasts for each model: {model_name: {ticker: forecast_array}}
    future_forecasts_by_model: Dict[str, Dict[str, np.ndarray]] = {
        "Naive": {},
        "ARIMA": {},
        "LSTM": {},
        "N-HiTS": {}
    }

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
        
        # Collect future forecasts for all models
        for model_name, forecast_vals in future_forecasts.items():
            if forecast_vals is not None and len(forecast_vals) > 0:
                future_forecasts_by_model[model_name][ticker] = forecast_vals

    # Step e & f: Analyze portfolio design, select top 5, allocate weights, and project profits for each model
    for model_name in ["Naive", "ARIMA", "LSTM", "N-HiTS"]:
        forecasts = future_forecasts_by_model[model_name]
        if not forecasts:
            logging.info("\n===== Skipping portfolio allocation for %s (no forecasts available) =====", model_name)
            continue
            
        logging.info("\n" + "="*80)
        logging.info(f"PORTFOLIO ALLOCATIONS & ANALYSIS FOR MODEL: {model_name.upper()}")
        logging.info("="*80)

        # Method 1: Return-Proportional Weighting
        logging.info("\n----- Running Return-Proportional Portfolio Allocation -----")
        ret_portfolio_df = analyze_and_allocate_portfolio(
            all_stocks_data=all_stocks_data,
            tickers=TICKERS,
            future_forecasts_all=forecasts,
            investment_idr=1_000_000_000_000.0,  # 1 Trillion IDR
            top_n=5,
            weighting_method="return_proportional",
            model_name=model_name
        )

        # Method 2: Markowitz Max Sharpe Ratio Weighting
        logging.info("\n----- Running Markowitz Max Sharpe Ratio Portfolio Allocation -----")
        markowitz_portfolio_df = analyze_and_allocate_portfolio(
            all_stocks_data=all_stocks_data,
            tickers=TICKERS,
            future_forecasts_all=forecasts,
            investment_idr=1_000_000_000_000.0,  # 1 Trillion IDR
            top_n=5,
            weighting_method="markowitz",
            model_name=model_name
        )

        # Compare portfolio performance metrics
        if not ret_portfolio_df.empty:
            logging.info("\n===== %s Model: Return-Proportional Portfolio Metrics Summary =====", model_name)
            optimized_tickers = ret_portfolio_df["Ticker"].tolist()
            optimized_weights = ret_portfolio_df["Weight"].tolist()
            compute_portfolio_metrics(all_stocks_data, optimized_tickers, optimized_weights)

        if not markowitz_portfolio_df.empty:
            logging.info("\n===== %s Model: Markowitz Portfolio Metrics Summary =====", model_name)
            optimized_tickers = markowitz_portfolio_df["Ticker"].tolist()
            optimized_weights = markowitz_portfolio_df["Weight"].tolist()
            compute_portfolio_metrics(all_stocks_data, optimized_tickers, optimized_weights)


if __name__ == "__main__":
    main()