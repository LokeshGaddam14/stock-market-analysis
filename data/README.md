# Data Directory

## Overview
This directory contains historical stock market data and trading signals used for the Stock Market Analysis project.

## Dataset Information

### Data Sources
- **Yahoo Finance API**: Historical OHLCV (Open, High, Low, Close, Volume) data
- **Alternative**: CSV files with pre-downloaded data
- **Format**: CSV files or JSON
- **Time Period**: Configurable (typical: 5+ years of daily data)

### Data Structure
```
data/
├── raw/
│   ├── stock_prices/       # Historical OHLCV data by ticker
│   └── market_indices/     # Index data (S&P 500, etc.)
├── processed/
│   └── technical_indicators/  # Calculated indicators
└── backtest/
    └── results/            # Backtest output files
```

## Data Format

### OHLCV Data Columns
- **Date**: Trading date
- **Open**: Opening price
- **High**: Highest price of the day
- **Low**: Lowest price of the day
- **Close**: Closing price
- **Volume**: Trading volume
- **Adj Close**: Adjusted closing price

## Technical Indicators
Calculated indicators include:
- Simple Moving Averages (SMA): 20, 50, 200-day
- Exponential Moving Averages (EMA): 12, 26-day
- MACD (Moving Average Convergence Divergence)
- RSI (Relative Strength Index)
- Bollinger Bands

## Data Preprocessing
- Missing values: Forward fill
- Outliers: Identified using IQR method
- Normalization: Min-Max scaling for ML models

## Usage
The `TradingStrategy` class loads and processes data from this directory.

## Data Sources & Attribution
Historical data courtesy of Yahoo Finance. For live data feeds, configure API credentials.
