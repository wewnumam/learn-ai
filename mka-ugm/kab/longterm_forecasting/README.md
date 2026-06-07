# Long-Term Stock Forecasting and Analysis

This workspace contains an exploratory forecasting script focused on Indonesian stock tickers listed on IDX (`.JK`) using historical OHLCV data from Yahoo Finance. The main code is in `untitled0.py`, and the project follows a CRISP-DM-inspired forecasting pipeline.

## Project Overview

The script performs the following tasks:
- Download historical price and volume data for a set of tickers.
- Compute technical indicators and price signals for a target stock.
- Build datasets for time-series forecasting.
- Train and compare multiple forecasting approaches: naive baseline, ARIMA, LSTM, and optional N-HiTS.
- Evaluate model performance using regression metrics and directional accuracy.
- Plot exploratory analysis, risk/return summaries, and forecast comparisons.

## Code Configuration

The main configuration values are defined at the top of `untitled0.py`:

- `TICKERS`: list of tickers to analyze.
- `TARGET_TICKER`: the ticker used for model training and forecasting.
- `LOOK_BACK`: number of lag days used to create LSTM input sequences.
- `FORECAST_HORIZON`: number of days to forecast into the future.
- `RISK_FREE_RATE`: annual risk-free rate used for portfolio Sharpe ratio calculations.

### Example default values

- `TICKERS = ["GOTO.JK", "MTDL.JK", "DCII.JK", "MLPT.JK", "BUKA.JK"]`
- `TARGET_TICKER = "GOTO.JK"`
- `LOOK_BACK = 60`
- `FORECAST_HORIZON = 30`
- `RISK_FREE_RATE = 0.05`

## Dependencies

The dependencies are listed in `requirements.txt`. Install them with:

```bash
pip install -r requirements.txt
```

Optional packages allow additional modeling functionality:
- `ta`: calculates technical indicators using the `ta` library.
- `pmdarima`: enables ARIMA model training.
- `neuralforecast`: enables the N-HiTS deep learning model.
- `pytorch-lightning`: used by `neuralforecast` callbacks if available.

The script gracefully degrades if optional packages are missing.

## Data and Feature Preparation

### Data source

- Historical stock OHLCV data is downloaded from Yahoo Finance using `yfinance`.
- The default period is 5 years.

### Feature engineering

The script computes technical indicators for the target stock:
- Moving averages: `MA50`, `MA200`
- Exponential moving averages: `EMA12`, `EMA26`
- MACD + signal line
- RSI
- Bollinger Bands (`BBL`, `BBH`)
- Support and resistance levels

When `ta` is installed, the script uses `ta` functions to compute these indicators. Otherwise, it falls back to pandas-based calculations.

### Dataset creation

The LSTM dataset is built by `create_lstm_dataset()`, which uses a sliding window over the feature matrix.
- Input features include price and technical indicators.
- The first output feature is the closing price.

The data is scaled using `MinMaxScaler` before entering the LSTM.

## Modeling and Forecasting

The supported forecasting models are:

- `Naive`: last observed value repeated for the horizon.
- `ARIMA`: auto-fitted ARIMA model via `pmdarima` if installed.
- `LSTM`: a two-layer recurrent neural network built with TensorFlow/Keras.
- `N-HiTS`: a deep forecasting model via `neuralforecast`, optional.

### LSTM model

The LSTM architecture uses:
- 2 LSTM layers
- dropout regularization
- a small dense output head
- Adam optimizer with mean squared error loss

Hyperparameters are currently set with fixed values in `tune_lstm_hyperparameters()` and `compare_models()`.

### Forecast comparison

`compare_models()` trains models on a training split and evaluates them on a test split.
It computes:
- MAE
- RMSE
- MAPE
- R2
- DSTAT (directional accuracy)

It also plots model predictions versus actual close prices and displays the best performing model for each metric.

## Exploratory Analysis and Visualization

The code includes visualization functions for:
- close price trends and all-time highs
- volume trends and summary statistics
- volatility and return histograms
- correlation heatmaps across tickers
- risk/return scatter plots
- buy/sell signals using moving average crosses and RSI

These functions are useful for data understanding and supporting the model evaluation stage.

## CRISP-DM Process in this Project

This code follows the CRISP-DM framework loosely as described below.

### 1. Business Understanding

- Goal: forecast long-term stock price movement for an IDX-listed target ticker.
- Focus: compare traditional and deep learning methods while understanding risk and technical signals.
- Success criteria: obtain a model with lower forecast error and good directional accuracy.

### 2. Data Understanding

- Data is sourced from Yahoo Finance.
- The code provides exploratory plots and summary statistics.
- It checks correlations, volatility, and return distributions across tickers.

### 3. Data Preparation

- Missing data is handled by dropping rows after indicator calculation.
- Technical indicators and support/resistance levels are engineered.
- The LSTM dataset is created using a rolling look-back window.
- Features are scaled using `MinMaxScaler`.

### 4. Modeling

- Models are trained and compared using the same prepared dataset.
- The LSTM is the primary neural model.
- ARIMA and N-HiTS are used as complementary baselines when available.

### 5. Evaluation

- Forecasts are compared on test data using regression metrics.
- Directional accuracy is computed to measure whether the model captures the direction of price movement.
- Visual comparisons are plotted for actuals vs predictions.

### 6. Deployment

- This repository is exploratory and currently meant for interactive analysis.
- The script can be extended into a reproducible notebook or modular pipeline for production.

## How to Use

This script appears to be designed for interactive execution, e.g. in a Jupyter notebook or Python console.

A reasonable workflow is:
1. Install dependencies.
2. Open `untitled0.py` in an interactive Python environment.
3. Import or run the functions in order:
   - `download_data()`
   - `compute_technical_indicators()`
   - visualization functions
   - `compare_models()`
4. Review plots and compare forecasting metrics.

## Notes

- `untitled0.py` is exploratory code; it is not packaged as a reusable module yet.
- If you want a reproducible pipeline, consider adding a `main()` function or converting this into a notebook.
- Optional packages improve modeling coverage but are not required for the basic LSTM pipeline.

## Recommended Improvements

- Add a `main()` entry point or notebook wrapper.
- Add command-line arguments for ticker, horizon, and look-back.
- Implement true hyperparameter tuning instead of fixed parameter dictionaries.
- Add stronger data cleaning and missing-value handling.
- Save model outputs and forecast results to disk.
