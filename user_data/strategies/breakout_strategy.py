# improved_breakout_strategy.py
# Drop into user_data/strategies/
# Strategy: Dynamic breakout + trend/range handling with risk management
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
import talib.abstract as ta
import numpy as np


class ImprovedBreakoutStrategy(IStrategy):
    """
    Improved Breakout Strategy (dynamic):
    - Uses multi-indicator filters: EMA trend, RSI, MACD, ADX, Volume
    - Handles four market modes:
        * Strong Uptrend: only long on confirmed breakouts above EMA
        * Strong Downtrend: only short on confirmed breakdowns below EMA
        * Ranging: use RSI reversal signals
        * Consolidation/weak trend: wait for ADX/RSI confirmation
    - Dynamic breakout: uses recent volatility (ATR) to define breakout threshold
    - Stoploss, take-profit levels, trailing stop available
    """

    # --- strategy config ---
    timeframe = "5m"                 # timeframe to run on (can override in config)
    minimal_roi = {"0": 0.03, "30": 0.02, "120": 0.01}
    stoploss = -0.05                 # default stoploss -5% (change in config if you want)
    trailing_stop = True
    trailing_stop_positive = 0.01    # start trailing when 1% in profit
    trailing_stop_positive_offset = 0.02
    trailing_only_offset_is_reached = True

    # parameters you can tune
    ema_fast = 20
    ema_slow = 50
    rsi_period = 14
    rsi_oversold = 30
    rsi_overbought = 70
    adx_period = 14
    adx_threshold = 25               # ADX > 25 considered trending
    macd_fast = 12
    macd_slow = 26
    macd_signal = 9
    atr_period = 14                  # use ATR for dynamic breakout threshold
    min_volume_multiplier = 0.8      # require volume >= 0.8 * vol_ma20

    # safety: minimum candles before using indicators
    startup_candle_count = 200

    # ---------------------------
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # EMAs
        dataframe[f'ema_{self.ema_fast}'] = ta.EMA(dataframe, timeperiod=self.ema_fast)
        dataframe[f'ema_{self.ema_slow}'] = ta.EMA(dataframe, timeperiod=self.ema_slow)

        # RSI
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=self.rsi_period)

        # MACD
        macd = ta.MACD(dataframe, fastperiod=self.macd_fast, slowperiod=self.macd_slow, signalperiod=self.macd_signal)
        dataframe['macd'] = macd['macd']
        dataframe['macd_signal'] = macd['macdsignal']
        dataframe['macd_hist'] = macd['macdhist']

        # ADX
        dataframe['adx'] = ta.ADX(dataframe, timeperiod=self.adx_period)

        # ATR for breakout threshold
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=self.atr_period)

        # Volume moving average
        dataframe['vol_ma20'] = dataframe['volume'].rolling(20).mean()

        # Recent highs / lows for breakout reference (lookback window)
        lookback = 20
        dataframe['high_recent'] = dataframe['high'].rolling(lookback).max()
        dataframe['low_recent'] = dataframe['low'].rolling(lookback).min()

        # A simple momentum measure: close - ema(slow)
        dataframe['price_above_ema_slow'] = dataframe['close'] - dataframe[f'ema_{self.ema_slow}']

        return dataframe

    # ---------------------------
    def _market_mode(self, row) -> str:
        """
        Decide market mode based on ADX and EMA slope:
        - 'strong_up', 'strong_down', 'range', 'consolidation'
        """
        # We'll use row values provided from dataframe when vectorized
        adx = row['adx']
        price_vs_ema = row['price_above_ema_slow']

        if adx >= self.adx_threshold:
            # trending market: upward or downward depending on price vs EMA
            return 'strong_up' if price_vs_ema > 0 else 'strong_down'
        else:
            # Non trending -> check RSI dispersion / range
            return 'range' if (row['rsi'] < 60 and row['rsi'] > 40) else 'consolidation'

    # ---------------------------
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Create entry signals. We'll implement vectorized logic:
        - dynamic breakout threshold = ATR * multiplier (e.g. 0.7)
        - require volume check and indicator confirmations
        """

        # no entries until enough candles present
        dataframe.loc[:, 'enter'] = 0
        if dataframe.shape[0] < self.startup_candle_count:
            return dataframe

        # dynamic breakout multiplier
        breakout_multiplier = 0.7

        # conditions for potential long breakout
        long_breakout = (
            (dataframe['close'] > dataframe['high_recent'] + dataframe['atr'] * breakout_multiplier) &       # price broke above recent high + ATR*mult
            (dataframe['macd'] > dataframe['macd_signal']) &                                                  # MACD bullish
            (dataframe['close'] > dataframe[f'ema_{self.ema_slow}']) &                                        # price above slow EMA
            (dataframe['volume'] >= dataframe['vol_ma20'] * self.min_volume_multiplier) &                      # volume not too low
            (dataframe['adx'] >= self.adx_threshold)                                                          # trending confirmed
        )

        # conditions for potential short breakout
        short_breakout = (
            (dataframe['close'] < dataframe['low_recent'] - dataframe['atr'] * breakout_multiplier) &
            (dataframe['macd'] < dataframe['macd_signal']) &
            (dataframe['close'] < dataframe[f'ema_{self.ema_slow}']) &
            (dataframe['volume'] >= dataframe['vol_ma20'] * self.min_volume_multiplier) &
            (dataframe['adx'] >= self.adx_threshold)
        )

        # conditions for range reversal (RSI oversold/overbought)
        range_long = (
            (dataframe['adx'] < self.adx_threshold) &
            (dataframe['rsi'] <= self.rsi_oversold) &
            (dataframe['close'] > dataframe[f'ema_{self.ema_fast}'])   # short term confirmation
        )
        range_short = (
            (dataframe['adx'] < self.adx_threshold) &
            (dataframe['rsi'] >= self.rsi_overbought) &
            (dataframe['close'] < dataframe[f'ema_{self.ema_fast}'])
        )

        # Consolidation mode: require stronger RSI extremes to trade
        consolidation_long = (
            (dataframe['adx'] < self.adx_threshold) &
            (dataframe['rsi'] <= (self.rsi_oversold + 5)) &  # slightly looser
            (dataframe['macd'] > dataframe['macd_signal'])
        )
        consolidation_short = (
            (dataframe['adx'] < self.adx_threshold) &
            (dataframe['rsi'] >= (self.rsi_overbought - 5)) &
            (dataframe['macd'] < dataframe['macd_signal'])
        )

        # Combine depending on mode
        dataframe.loc[
            (long_breakout) | (range_long) | (consolidation_long),
            'enter'] = 1

        dataframe.loc[
            (short_breakout) | (range_short) | (consolidation_short),
            'enter'] = -1

        return dataframe

    # ---------------------------
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Simple exits:
        - exit when opposite signal triggers OR RSI reverts strongly OR price crosses EMA slow.
        - we also allow ROI and stoploss configured by freqtrade settings
        """
        dataframe.loc[:, 'exit'] = 0

        # Exit long if price falls back below EMA slow or RSI becomes overbought
        exit_long = (
            (dataframe['close'] < dataframe[f'ema_{self.ema_slow}']) |
            (dataframe['rsi'] >= self.rsi_overbought)
        )

        # Exit short if price rises above EMA slow or RSI becomes oversold
        exit_short = (
            (dataframe['close'] > dataframe[f'ema_{self.ema_slow}']) |
            (dataframe['rsi'] <= self.rsi_oversold)
        )

        dataframe.loc[exit_long, 'exit'] = 1
        dataframe.loc[exit_short, 'exit'] = -1

        return dataframe
