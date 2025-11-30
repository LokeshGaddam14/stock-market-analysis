"""
Stock Market Analysis - Trading Strategy Module

This module contains the TradingStrategy class for implementing technical analysis
and generating trading signals based on various technical indicators.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, List
import warnings
warnings.filterwarnings('ignore')


class TradingStrategy:
    """
    A comprehensive trading strategy implementation using technical analysis.
    
    This class implements various technical indicators and signal generation mechanisms
    for algorithmic trading, including moving averages, RSI, MACD, and Bollinger Bands.
    """
    
    def __init__(self, short_window: int = 20, long_window: int = 50):
        """
        Initialize the trading strategy.
        
        Args:
            short_window (int): Short-term moving average window. Default: 20
            long_window (int): Long-term moving average window. Default: 50
        """
        self.short_window = short_window
        self.long_window = long_window
        self.signals = None
        self.portfolio_value = None
        
    def calculate_moving_averages(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate short and long-term moving averages.
        
        Args:
            data (pd.DataFrame): Price data with 'Close' column
            
        Returns:
            pd.DataFrame: Data with SMA_short and SMA_long columns
        """
        data = data.copy()
        data['SMA_short'] = data['Close'].rolling(window=self.short_window).mean()
        data['SMA_long'] = data['Close'].rolling(window=self.long_window).mean()
        return data
    
    def calculate_rsi(self, data: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        Calculate Relative Strength Index (RSI).
        
        Args:
            data (pd.DataFrame): Price data with 'Close' column
            period (int): RSI period. Default: 14
            
        Returns:
            pd.DataFrame: Data with RSI column
        """
        data = data.copy()
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))
        return data
    
    def calculate_macd(self, data: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        """
        Calculate MACD (Moving Average Convergence Divergence).
        
        Args:
            data (pd.DataFrame): Price data with 'Close' column
            fast (int): Fast EMA period. Default: 12
            slow (int): Slow EMA period. Default: 26
            signal (int): Signal line period. Default: 9
            
        Returns:
            pd.DataFrame: Data with MACD and signal line columns
        """
        data = data.copy()
        data['EMA_fast'] = data['Close'].ewm(span=fast).mean()
        data['EMA_slow'] = data['Close'].ewm(span=slow).mean()
        data['MACD'] = data['EMA_fast'] - data['EMA_slow']
        data['Signal_line'] = data['MACD'].ewm(span=signal).mean()
        data['MACD_hist'] = data['MACD'] - data['Signal_line']
        return data
    
    def calculate_bollinger_bands(self, data: pd.DataFrame, period: int = 20, std_dev: float = 2.0) -> pd.DataFrame:
        """
        Calculate Bollinger Bands.
        
        Args:
            data (pd.DataFrame): Price data with 'Close' column
            period (int): Moving average period. Default: 20
            std_dev (float): Number of standard deviations. Default: 2.0
            
        Returns:
            pd.DataFrame: Data with Bollinger Bands columns
        """
        data = data.copy()
        data['BB_middle'] = data['Close'].rolling(window=period).mean()
        bb_std = data['Close'].rolling(window=period).std()
        data['BB_upper'] = data['BB_middle'] + (bb_std * std_dev)
        data['BB_lower'] = data['BB_middle'] - (bb_std * std_dev)
        return data
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on technical indicators.
        
        Args:
            data (pd.DataFrame): Price data with technical indicators
            
        Returns:
            pd.DataFrame: Data with trading signals
        """
        data = data.copy()
        data['Signal'] = 0
        data['Position'] = 0
        
        # SMA crossover strategy
        data.loc[data['SMA_short'] > data['SMA_long'], 'Signal'] = 1
        data.loc[data['SMA_short'] <= data['SMA_long'], 'Signal'] = -1
        
        # Position changes when signal changes
        data['Position'] = data['Signal'].diff()
        
        self.signals = data
        return data
    
    def backtest(self, data: pd.DataFrame, initial_capital: float = 100000) -> Dict:
        """
        Backtest the trading strategy.
        
        Args:
            data (pd.DataFrame): Price data with signals
            initial_capital (float): Initial investment capital. Default: 100000
            
        Returns:
            Dict: Backtest results including returns, Sharpe ratio, max drawdown
        """
        data = data.copy()
        data['Portfolio_value'] = initial_capital
        data['Returns'] = data['Close'].pct_change()
        data['Strategy_returns'] = data['Signal'].shift(1) * data['Returns']
        data['Cumulative_returns'] = (1 + data['Strategy_returns']).cumprod()
        data['Portfolio_value'] = initial_capital * data['Cumulative_returns']
        
        self.portfolio_value = data['Portfolio_value']
        
        total_return = (data['Portfolio_value'].iloc[-1] - initial_capital) / initial_capital
        annual_return = data['Strategy_returns'].mean() * 252
        annual_volatility = data['Strategy_returns'].std() * np.sqrt(252)
        sharpe_ratio = annual_return / annual_volatility if annual_volatility > 0 else 0
        
        # Calculate maximum drawdown
        cummax = data['Portfolio_value'].expanding().max()
        drawdown = (data['Portfolio_value'] - cummax) / cummax
        max_drawdown = drawdown.min()
        
        results = {
            'total_return': total_return,
            'annual_return': annual_return,
            'annual_volatility': annual_volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'final_portfolio_value': data['Portfolio_value'].iloc[-1]
        }
        
        return results
    
    def get_trading_signals(self) -> pd.DataFrame:
        """
        Get the generated trading signals.
        
        Returns:
            pd.DataFrame: Signals dataframe
        """
        return self.signals
    
    def get_performance_metrics(self) -> Dict:
        """
        Get performance metrics from backtesting.
        
        Returns:
            Dict: Performance metrics
        """
        if self.portfolio_value is None:
            raise ValueError("Backtest not run yet. Call backtest() first.")
        return {
            'portfolio_values': self.portfolio_value.values,
            'final_value': self.portfolio_value.iloc[-1]
        }
