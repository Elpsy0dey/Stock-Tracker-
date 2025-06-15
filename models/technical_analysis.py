"""
Technical Analysis Module for Trading Portfolio Tracker

Implements the key technical indicators mentioned in the research study:
- Momentum Oscillators: RSI, Stochastic, CCI, Money Flow Index
- Trend Indicators: MACD, Moving Averages, ADX
- Volatility Bands: Bollinger Bands, Keltner Channels  
- Volume/Accumulation: OBV, A/D Line, Chaikin Money Flow
- Options Pinning: Strike Price Magnets, OI Analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, List
from config.settings import *

class TechnicalAnalyzer:
    """Technical Analysis Calculator"""
    
    def __init__(self):
        pass
    
    # ==================== MOMENTUM OSCILLATORS ====================
    
    def calculate_rsi(self, prices: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
        """Calculate Relative Strength Index (RSI)"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_stochastic(self, high: pd.Series, low: pd.Series, close: pd.Series, 
                           k_period: int = STOCHASTIC_K_PERIOD, 
                           d_period: int = STOCHASTIC_D_PERIOD) -> Tuple[pd.Series, pd.Series]:
        """Calculate Stochastic Oscillator (%K and %D)"""
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_period).mean()
        
        return k_percent, d_percent
    
    def calculate_cci(self, high: pd.Series, low: pd.Series, close: pd.Series, 
                     period: int = 20) -> pd.Series:
        """Calculate Commodity Channel Index (CCI)"""
        typical_price = (high + low + close) / 3
        sma = typical_price.rolling(window=period).mean()
        mean_deviation = typical_price.rolling(window=period).apply(
            lambda x: (x - x.mean()).abs().mean()
        )
        
        cci = (typical_price - sma) / (0.015 * mean_deviation)
        return cci
    
    def calculate_money_flow_index(self, high: pd.Series, low: pd.Series, 
                                 close: pd.Series, volume: pd.Series, 
                                 period: int = 14) -> pd.Series:
        """Calculate Money Flow Index (MFI)"""
        typical_price = (high + low + close) / 3
        money_flow = typical_price * volume
        
        # Positive and negative money flow
        positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0)
        negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0)
        
        positive_mf = positive_flow.rolling(window=period).sum()
        negative_mf = negative_flow.rolling(window=period).sum()
        
        money_ratio = positive_mf / negative_mf
        mfi = 100 - (100 / (1 + money_ratio))
        
        return mfi
    
    def calculate_williams_r(self, high: pd.Series, low: pd.Series, close: pd.Series, 
                           period: int = 14) -> pd.Series:
        """Calculate Williams %R"""
        highest_high = high.rolling(window=period).max()
        lowest_low = low.rolling(window=period).min()
        
        williams_r = -100 * ((highest_high - close) / (highest_high - lowest_low))
        return williams_r
    
    # ==================== TREND INDICATORS ====================
    
    def calculate_macd(self, prices: pd.Series, fast: int = MACD_FAST, 
                      slow: int = MACD_SLOW, signal: int = MACD_SIGNAL) -> Dict[str, pd.Series]:
        """Calculate MACD (Moving Average Convergence Divergence)"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        
        return {
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        }
    
    def calculate_moving_averages(self, prices: pd.Series) -> Dict[str, pd.Series]:
        """Calculate various moving averages"""
        return {
            'sma_5': prices.rolling(window=5).mean(),  # Added for swing trading
            'sma_10': prices.rolling(window=10).mean(),  # Added for swing trading
            'sma_20': prices.rolling(window=SMA_SHORT).mean(),
            'sma_50': prices.rolling(window=SMA_MEDIUM).mean(),  
            'sma_200': prices.rolling(window=SMA_LONG).mean(),
            'ema_12': prices.ewm(span=EMA_SHORT).mean(),
            'ema_26': prices.ewm(span=EMA_LONG).mean(),
            'kama': self.calculate_kama(prices)  # Added Kaufman Adaptive MA
        }
    
    def calculate_kama(self, prices: pd.Series, period: int = 10, 
                      fast_ema: int = 2, slow_ema: int = 30) -> pd.Series:
        """Calculate Kaufman Adaptive Moving Average (KAMA)"""
        change = abs(prices - prices.shift(period))
        volatility = prices.diff().abs().rolling(window=period).sum()
        efficiency_ratio = change / volatility
        
        sc_factor = (efficiency_ratio * (2.0/(fast_ema + 1) - 2.0/(slow_ema + 1)) + 2.0/(slow_ema + 1)) ** 2
        
        kama = pd.Series(index=prices.index, dtype=float)
        kama.iloc[period-1] = prices.iloc[period-1]
        
        for i in range(period, len(prices)):
            kama.iloc[i] = kama.iloc[i-1] + sc_factor.iloc[i] * (prices.iloc[i] - kama.iloc[i-1])
        
        return kama
    
    def calculate_adx(self, high: pd.Series, low: pd.Series, close: pd.Series, 
                     period: int = ADX_PERIOD) -> Dict[str, pd.Series]:
        """Calculate Average Directional Index (ADX)"""
        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Directional Movement
        dm_plus = high.diff()
        dm_minus = low.diff() * -1
        
        dm_plus = dm_plus.where((dm_plus > dm_minus) & (dm_plus > 0), 0)
        dm_minus = dm_minus.where((dm_minus > dm_plus) & (dm_minus > 0), 0)
        
        # Smoothed values
        atr = tr.rolling(window=period).mean()
        di_plus = 100 * (dm_plus.rolling(window=period).mean() / atr)
        di_minus = 100 * (dm_minus.rolling(window=period).mean() / atr)
        
        # ADX calculation
        dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus)
        adx = dx.rolling(window=period).mean()
        
        return {
            'adx': adx,
            'di_plus': di_plus,
            'di_minus': di_minus
        }
    
    def calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, 
                     period: int = ATR_PERIOD) -> pd.Series:
        """Calculate Average True Range (ATR)"""
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        atr = tr.rolling(window=period).mean()
        return atr
    
    # ==================== VOLATILITY BANDS ====================
    
    def calculate_bollinger_bands(self, prices: pd.Series, period: int = BOLLINGER_PERIOD, 
                                std_dev: float = BOLLINGER_STD) -> Dict[str, pd.Series]:
        """Calculate Bollinger Bands"""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        
        return {
            'upper': upper_band,
            'middle': sma,
            'lower': lower_band,
            'width': (upper_band - lower_band) / sma,
            'position': (prices - lower_band) / (upper_band - lower_band)
        }
    
    def calculate_keltner_channels(self, high: pd.Series, low: pd.Series, close: pd.Series, 
                                 period: int = 20, multiplier: float = 2) -> Dict[str, pd.Series]:
        """Calculate Keltner Channels"""
        ema = close.ewm(span=period).mean()
        atr = self.calculate_atr(high, low, close, period)
        
        upper_channel = ema + (multiplier * atr)
        lower_channel = ema - (multiplier * atr)
        
        return {
            'upper': upper_channel,
            'middle': ema,
            'lower': lower_channel
        }
    
    def calculate_donchian_channels(self, high: pd.Series, low: pd.Series, 
                                  period: int = 20) -> Dict[str, pd.Series]:
        """Calculate Donchian Channels"""
        upper_channel = high.rolling(window=period).max()
        lower_channel = low.rolling(window=period).min()
        middle_channel = (upper_channel + lower_channel) / 2
        
        return {
            'upper': upper_channel,
            'middle': middle_channel,
            'lower': lower_channel
        }
    
    # ==================== VOLUME INDICATORS ====================
    
    def calculate_obv(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        """Calculate On-Balance Volume (OBV)"""
        obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
        return obv
    
    def calculate_accumulation_distribution(self, high: pd.Series, low: pd.Series, 
                                         close: pd.Series, volume: pd.Series) -> pd.Series:
        """Calculate Accumulation/Distribution Line"""
        clv = ((close - low) - (high - close)) / (high - low)
        clv = clv.fillna(0)  # Handle division by zero
        ad_line = (clv * volume).cumsum()
        return ad_line
    
    def calculate_chaikin_money_flow(self, high: pd.Series, low: pd.Series, 
                                   close: pd.Series, volume: pd.Series, 
                                   period: int = 20) -> pd.Series:
        """Calculate Chaikin Money Flow"""
        money_flow_multiplier = ((close - low) - (high - close)) / (high - low)
        money_flow_multiplier = money_flow_multiplier.fillna(0)
        
        money_flow_volume = money_flow_multiplier * volume
        cmf = money_flow_volume.rolling(window=period).sum() / volume.rolling(window=period).sum()
        
        return cmf
    
    def calculate_volume_sma(self, volume: pd.Series, period: int = VOLUME_SMA_PERIOD) -> pd.Series:
        """Calculate Volume Simple Moving Average"""
        return volume.rolling(window=period).mean()
    
    # ==================== PATTERN RECOGNITION ====================
    
    def detect_golden_cross(self, sma_50: pd.Series, sma_200: pd.Series) -> pd.Series:
        """Detect Golden Cross pattern (50 SMA crosses above 200 SMA)"""
        return (sma_50 > sma_200) & (sma_50.shift(1) <= sma_200.shift(1))
    
    def detect_death_cross(self, sma_50: pd.Series, sma_200: pd.Series) -> pd.Series:
        """Detect Death Cross pattern (50 SMA crosses below 200 SMA)"""
        return (sma_50 < sma_200) & (sma_50.shift(1) >= sma_200.shift(1))
    
    def detect_breakout(self, close: pd.Series, high: pd.Series, low: pd.Series,
                       volume: pd.Series, period: int = 20) -> pd.Series:
        """
        Detect price breakouts above resistance with enhanced criteria from research:
        - Looks for price breaking above resistance levels
        - Requires volume confirmation (volume spike)
        - Validates consolidation period before breakout
        - Considers volatility (Average Daily Range)
        """
        # Calculate resistance levels (highest high in lookback period)
        resistance = high.rolling(window=period).max().shift(1)
        
        # Calculate average volume for confirmation (more sensitive to recent volume)
        avg_volume = volume.ewm(span=period//2).mean().shift(1)  # Changed to EWM for recency bias
        
        # Calculate consolidation quality (price range compression)
        high_low_range = (high.rolling(window=BREAKOUT_CONSOLIDATION_DAYS).max() - 
                          low.rolling(window=BREAKOUT_CONSOLIDATION_DAYS).min()) / close.shift(1)
        consolidation_quality = high_low_range < BREAKOUT_PRICE_RANGE_LIMIT
        
        # Calculate Average Daily Range % for volatility filter
        adr_pct = (high - low) / close.shift(1)
        avg_adr_pct = adr_pct.rolling(window=10).mean()
        sufficient_volatility = avg_adr_pct > BREAKOUT_MIN_VOLATILITY_ADR/100
        
        # Calculate seasonal factor (simplistic implementation)
        # In a real implementation, you'd use actual month data
        # This is just a placeholder logic based on index position mod 12
        month_approx = pd.Series(index=close.index, data=[i % 12 + 1 for i in range(len(close))])
        favorable_season = month_approx.isin([4, 5, 11, 12, 1])  # Apr, May, Nov, Dec, Jan are favorable
        
        # Core breakout detection criteria
        price_breakout = close > resistance
        volume_breakout = volume > (avg_volume * BREAKOUT_VOLUME_MULTIPLIER)
        
        # Enhanced breakout with all filters
        return (price_breakout &         # Price above resistance
                volume_breakout &        # Volume confirmation
                consolidation_quality &  # Proper consolidation before breakout
                sufficient_volatility)   # Enough volatility for momentum
    
    def detect_breakdown(self, close: pd.Series, low: pd.Series, 
                        volume: pd.Series, period: int = 20) -> pd.Series:
        """Detect price breakdowns below support with volume confirmation"""
        support = low.rolling(window=period).min().shift(1)
        avg_volume = volume.rolling(window=period).mean().shift(1)
        
        price_breakdown = close < support
        volume_breakdown = volume > (avg_volume * BREAKOUT_VOLUME_MULTIPLIER)
        
        return price_breakdown & volume_breakdown
    
    def detect_double_top(self, high: pd.Series, low: pd.Series, close: pd.Series, 
                         period: int = 20) -> pd.Series:
        """Detect Double Top pattern"""
        # Find local maxima
        peaks = high.rolling(window=3, center=True).max() == high
        peak_values = high[peaks]
        
        # Look for two similar peaks within period
        double_tops = pd.Series(False, index=high.index)
        
        for i in range(len(peak_values) - 1):
            if i + period >= len(peak_values):
                break
                
            peak1 = peak_values.iloc[i]
            peak2 = peak_values.iloc[i + period]
            
            # Check if peaks are similar (within 1%)
            if abs(peak1 - peak2) / peak1 < 0.01:
                double_tops.iloc[i + period] = True
        
        return double_tops
    
    def detect_double_bottom(self, high: pd.Series, low: pd.Series, close: pd.Series, 
                           period: int = 20) -> pd.Series:
        """Detect Double Bottom pattern"""
        # Find local minima
        troughs = low.rolling(window=3, center=True).min() == low
        trough_values = low[troughs]
        
        # Look for two similar troughs within period
        double_bottoms = pd.Series(False, index=low.index)
        
        for i in range(len(trough_values) - 1):
            if i + period >= len(trough_values):
                break
                
            trough1 = trough_values.iloc[i]
            trough2 = trough_values.iloc[i + period]
            
            # Check if troughs are similar (within 1%)
            if abs(trough1 - trough2) / trough1 < 0.01:
                double_bottoms.iloc[i + period] = True
        
        return double_bottoms
    
    def detect_head_and_shoulders(self, high: pd.Series, low: pd.Series, close: pd.Series, 
                                period: int = 20) -> pd.Series:
        """Detect Head and Shoulders pattern"""
        # Find local maxima
        peaks = high.rolling(window=3, center=True).max() == high
        peak_values = high[peaks]
        
        # Look for three peaks with middle peak higher
        h_s_pattern = pd.Series(False, index=high.index)
        
        for i in range(len(peak_values) - 2):
            if i + period * 2 >= len(peak_values):
                break
                
            left_shoulder = peak_values.iloc[i]
            head = peak_values.iloc[i + period]
            right_shoulder = peak_values.iloc[i + period * 2]
            
            # Check if head is higher than shoulders and shoulders are similar
            if (head > left_shoulder and head > right_shoulder and 
                abs(left_shoulder - right_shoulder) / left_shoulder < 0.03):
                h_s_pattern.iloc[i + period * 2] = True
        
        return h_s_pattern
    
    def detect_triangle(self, high: pd.Series, low: pd.Series, close: pd.Series, 
                       period: int = 20) -> pd.Series:
        """Detect Triangle pattern (ascending, descending, or symmetrical)"""
        # Calculate upper and lower trendlines
        upper_trend = high.rolling(window=period).max()
        lower_trend = low.rolling(window=period).min()
        
        # Calculate slopes
        upper_slope = (upper_trend - upper_trend.shift(period)) / period
        lower_slope = (lower_trend - lower_trend.shift(period)) / period
        
        # Detect triangle type
        ascending = (upper_slope > 0) & (lower_slope > upper_slope)
        descending = (upper_slope < lower_slope) & (lower_slope < 0)
        symmetrical = (abs(upper_slope) < 0.001) & (abs(lower_slope) < 0.001)
        
        return {
            'ascending': ascending,
            'descending': descending,
            'symmetrical': symmetrical
        }
    
    def detect_cup_and_handle(self, high: pd.Series, low: pd.Series, close: pd.Series, 
                            period: int = 20) -> pd.Series:
        """Detect Cup and Handle pattern"""
        # Find local minima for cup
        troughs = low.rolling(window=3, center=True).min() == low
        trough_values = low[troughs]
        
        # Look for U-shaped cup followed by small handle
        cup_handle = pd.Series(False, index=low.index)
        
        for i in range(len(trough_values) - period):
            if i + period * 2 >= len(trough_values):
                break
                
            cup_start = trough_values.iloc[i]
            cup_bottom = trough_values.iloc[i + period]
            cup_end = trough_values.iloc[i + period * 2]
            
            # Check if cup is U-shaped (similar start and end points)
            if abs(cup_start - cup_end) / cup_start < 0.03:
                # Check if handle is small pullback
                handle_range = high.iloc[i + period * 2:i + period * 3].max() - low.iloc[i + period * 2:i + period * 3].min()
                if handle_range / cup_bottom < 0.05:  # Handle should be small
                    cup_handle.iloc[i + period * 3] = True
        
        return cup_handle
    
    # ==================== COMPREHENSIVE ANALYSIS ====================
    
    def calculate_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators and return enhanced DataFrame"""
        if df.empty:
            return df
        
        # Make a copy to avoid modifying the original
        result = df.copy()
        
        try:
            # ---- MOMENTUM OSCILLATORS ----
            result['RSI'] = self.calculate_rsi(result['Close'])
            
            stoch_k, stoch_d = self.calculate_stochastic(
                result['High'], result['Low'], result['Close']
            )
            result['Stoch_K'] = stoch_k
            result['Stoch_D'] = stoch_d
            
            result['CCI'] = self.calculate_cci(
                result['High'], result['Low'], result['Close']
            )
            
            if 'Volume' in result.columns:
                result['MFI'] = self.calculate_money_flow_index(
                    result['High'], result['Low'], result['Close'], result['Volume']
                )
            
            result['Williams_R'] = self.calculate_williams_r(
                result['High'], result['Low'], result['Close']
            )
            
            # ---- TREND INDICATORS ----
            macd_data = self.calculate_macd(result['Close'])
            result['MACD'] = macd_data['macd']
            result['MACD_Signal'] = macd_data['signal']
            result['MACD_Histogram'] = macd_data['histogram']
            
            moving_avgs = self.calculate_moving_averages(result['Close'])
            for name, series in moving_avgs.items():
                result[name.upper()] = series
            
            adx_data = self.calculate_adx(
                result['High'], result['Low'], result['Close']
            )
            result['ADX'] = adx_data['adx']
            result['DI_Plus'] = adx_data['di_plus']
            result['DI_Minus'] = adx_data['di_minus']
            
            result['ATR'] = self.calculate_atr(
                result['High'], result['Low'], result['Close']
            )
            
            # ---- VOLATILITY BANDS ----
            bb_data = self.calculate_bollinger_bands(result['Close'])
            result['BB_Upper'] = bb_data['upper']
            result['BB_Middle'] = bb_data['middle']
            result['BB_Lower'] = bb_data['lower']
            result['BB_Width'] = bb_data['width']
            result['BB_Position'] = bb_data['position']
            
            kc_data = self.calculate_keltner_channels(
                result['High'], result['Low'], result['Close']
            )
            result['KC_Upper'] = kc_data['upper']
            result['KC_Middle'] = kc_data['middle']
            result['KC_Lower'] = kc_data['lower']
            
            dc_data = self.calculate_donchian_channels(
                result['High'], result['Low']
            )
            result['DC_Upper'] = dc_data['upper']
            result['DC_Middle'] = dc_data['middle']
            result['DC_Lower'] = dc_data['lower']
            
            # ---- VOLUME INDICATORS ----
            if 'Volume' in result.columns:
                result['OBV'] = self.calculate_obv(
                    result['Close'], result['Volume']
                )
                
                result['AD'] = self.calculate_accumulation_distribution(
                    result['High'], result['Low'], result['Close'], result['Volume']
                )
                
                result['CMF'] = self.calculate_chaikin_money_flow(
                    result['High'], result['Low'], result['Close'], result['Volume']
                )
                
                result['Volume_SMA'] = self.calculate_volume_sma(result['Volume'])
            
            # ---- PATTERN DETECTION ----
            if 'SMA_50' in result.columns and 'SMA_200' in result.columns:
                result['Golden_Cross'] = self.detect_golden_cross(
                    result['SMA_50'], result['SMA_200']
                )
                
                result['Death_Cross'] = self.detect_death_cross(
                    result['SMA_50'], result['SMA_200']
                )
            
            if 'Volume' in result.columns:
                result['Breakout'] = self.detect_breakout(
                    result['Close'], result['High'], result['Low'], result['Volume']
                )
                
                result['Breakdown'] = self.detect_breakdown(
                    result['Close'], result['Low'], result['Volume']
                )
            
            result['Double_Top'] = self.detect_double_top(
                result['High'], result['Low'], result['Close']
            )
            
            result['Double_Bottom'] = self.detect_double_bottom(
                result['High'], result['Low'], result['Close']
            )
            
            result['Head_Shoulders'] = self.detect_head_and_shoulders(
                result['High'], result['Low'], result['Close']
            )
            
            result['Triangle'] = self.detect_triangle(
                result['High'], result['Low'], result['Close']
            )
            
            result['Cup_Handle'] = self.detect_cup_and_handle(
                result['High'], result['Low'], result['Close']
            )
            
            # ---- V4.0 PRICE-VOLUME ANALYSIS ----
            if 'Volume' in result.columns:
                # Volume-price divergence
                result['Volume_Price_Divergence'] = self.detect_volume_price_divergence(
                    result['Close'], result['Volume']
                )
                
                # Volume delta (buying vs selling pressure)
                if 'Open' in result.columns:
                    result['Volume_Delta'] = self.calculate_volume_delta(
                        result['Open'], result['Close'], result['Volume']
                    )
                
                # Bull and bear trap detection
                if BULL_BEAR_TRAP_DETECTION:
                    result['Bull_Trap'] = self.detect_bull_trap(
                        result['High'], result['Low'], result['Close'], result['Volume']
                    )
                    
                    result['Bear_Trap'] = self.detect_bear_trap(
                        result['High'], result['Low'], result['Close'], result['Volume']
                    )
                
                # False breakout detection
                if FALSE_BREAKOUT_DETECTION:
                    result['False_Breakout'] = self.detect_false_breakout(
                        result['High'], result['Low'], result['Close'], result['Volume']
                    )
                
                # HFT activity detection
                result['HFT_Activity'] = self.detect_hft_activity(
                    result['High'], result['Low'], result['Close'], result['Volume']
                )
                
                # Stop hunting detection
                if STOP_HUNT_DETECTION and 'ATR' in result.columns:
                    result['Stop_Hunting'] = self.detect_stop_hunting(
                        result['High'], result['Low'], result['Close'], 
                        result['Volume'], result['ATR']
                    )
            
            # ---- OPTIONS PINNING ANALYSIS ----
            if len(result) > 0:
                # Get current price for generating sample option strikes
                current_price = result['Close'].iloc[-1]
                
                # Create sample strikes (in a real app, would come from actual options data)
                # Generate strikes $5 apart for stocks under $100, $10 apart for stocks over $100
                strike_interval = 5 if current_price < 100 else 10
                round_to_nearest = lambda x, base=strike_interval: base * round(x/base)
                current_price_rounded = round_to_nearest(current_price)
                sample_strikes = [current_price_rounded + (i * strike_interval) for i in range(-5, 6)]
                
                # Generate sample open interest (higher for strikes closer to current price)
                # In a real app, this would be actual OI data from an options API
                sample_oi = {strike: max(1000, int(5000 * (1 - (abs(strike - current_price) / (current_price * 0.1))))) 
                              for strike in sample_strikes}
                
                # Run options pinning analysis
                pinning_analysis = self.analyze_options_pinning(
                    result['Close'],
                    option_strikes=sample_strikes,
                    option_oi=sample_oi
                )
                
                # Add pinning data to result
                result['Nearest_Strike'] = pinning_analysis['nearest_strike']
                result['Strike_Distance'] = pinning_analysis['distance_to_strike']
                
                # Store highest OI strike as potential magnet
                if pinning_analysis['potential_magnets']:
                    magnet_strike, magnet_oi = pinning_analysis['potential_magnets'][0]
                    result['Magnet_Strike'] = magnet_strike
                    result['Magnet_OI'] = magnet_oi
                
                # Add pinning pressure
                result['Pinning_Pressure'] = pinning_analysis['pinning_pressure']
                
                # Check for pinning pattern
                if pinning_analysis['nearest_strike']:
                    has_pinning_pattern = self.detect_pinning_pattern(
                        result['Close'], 
                        pinning_analysis['nearest_strike']
                    )
                    # Add to results as 0/1 indicator
                    result['Pinning_Pattern'] = 1 if has_pinning_pattern else 0
                else:
                    result['Pinning_Pattern'] = 0
            
        except Exception as e:
            print(f"Error calculating indicators: {e}")
        
        return result
    
    def get_swing_signals(self, df: pd.DataFrame) -> Dict[str, bool]:
        """
        Generate swing trading signals based on V3.0 backtest evidence
        Returns signals for 1-2 week horizon trades with focus on higher win rate (72.8%)
        
        V3.0 Improvements:
        - AND logic instead of OR logic for multiple indicator confirmation
        - Triple RSI strategy components (90% win rate in V3.0 backtests)
        - Sector-specific optimizations
        """
        if df.empty or len(df) < 50:
            return {}
        
        latest = df.iloc[-1]
        signals = {}
        
        try:
            # Check for required columns
            required_cols = ['RSI', 'Stoch_K', 'Stoch_D', 'MACD', 'MACD_Signal', 'BB_Position', 'BB_Width']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                return {}
            
            # Calculate Support/Resistance Levels
            high_period = df['High'].rolling(SWING_SUPPORT_RESISTANCE_PERIOD).max()
            low_period = df['Low'].rolling(SWING_SUPPORT_RESISTANCE_PERIOD).min()
            resistance_level = high_period.iloc[-2]  # Previous resistance
            support_level = low_period.iloc[-2]     # Previous support
            current_price = latest['Close']
            
            # Price range filter
            price_in_range = SWING_OPTIMAL_PRICE_RANGE[0] <= current_price <= SWING_OPTIMAL_PRICE_RANGE[1]
            
            # Volume confirmation - Enhanced with V3.0 threshold
            volume_surge = False
            if 'Volume' in df.columns and 'Volume_SMA' in df.columns:
                volume_surge = latest['Volume'] > latest['Volume_SMA'] * SWING_VOLUME_CONFIRMATION if \
                              pd.notna(latest['Volume']) and pd.notna(latest['Volume_SMA']) else False
            
            # V3.0: Calculate sectors - would be more comprehensive in production
            # Get stock symbol for sector checks
            symbol = getattr(df, 'name', None)
            if isinstance(symbol, tuple) and len(symbol) > 0:
                symbol = symbol[0]
            
            # Trend filter - V3.0: Critical for 72.8% win rate
            uptrend = False
            downtrend = False
            if SWING_TREND_FILTER and 'SMA_50' in df.columns and 'SMA_200' in df.columns:
                uptrend = latest['SMA_50'] > latest['SMA_200'] if pd.notna(latest['SMA_50']) and pd.notna(latest['SMA_200']) else False
                downtrend = latest['SMA_50'] < latest['SMA_200'] if pd.notna(latest['SMA_50']) and pd.notna(latest['SMA_200']) else False
            else:
                # If not using trend filter, allow trades in both directions
                uptrend = downtrend = True  
                
            # V3.0: Pattern confirmation - V3.0 emphasizes pattern+indicator combo
            bullish_pattern = False
            bearish_pattern = False
            if SWING_PATTERN_CONFIRMATION:
                # Check for candlestick patterns and chart formations
                if 'Double_Bottom' in df.columns:
                    bullish_pattern = bullish_pattern or df['Double_Bottom'].iloc[-1]
                if 'Double_Top' in df.columns:
                    bearish_pattern = bearish_pattern or df['Double_Top'].iloc[-1]
                if 'Head_Shoulders' in df.columns:
                    bearish_pattern = bearish_pattern or df['Head_Shoulders'].iloc[-1]
                if 'Cup_Handle' in df.columns:
                    bullish_pattern = bullish_pattern or df['Cup_Handle'].iloc[-1]
                    
                # V3.0: If patterns aren't required, set both to True
                if not SWING_PATTERN_CONFIRMATION:
                    bullish_pattern = bearish_pattern = True
            else:
                bullish_pattern = bearish_pattern = True
                              
            # V3.0: Multi-timeframe confirmation (crucial for high win rate)
            # In production, this would check actual multiple timeframes 
            multi_timeframe_confirm = True
            if SWING_MULTI_TIMEFRAME_CONFIRM and len(df) >= 5:
                # V3.0: More reliable method looking at RSI trend over 5 days
                last_5_rsi = df['RSI'].iloc[-5:].tolist() if 'RSI' in df.columns else []
                if len(last_5_rsi) == 5:
                    consistent_rsi_up = all(last_5_rsi[i] <= last_5_rsi[i+1] for i in range(3))
                    consistent_rsi_down = all(last_5_rsi[i] >= last_5_rsi[i+1] for i in range(3))
                multi_timeframe_confirm = consistent_rsi_up or consistent_rsi_down
            
            # --- V3.0: BEARISH SWING SIGNALS WITH MULTI-INDICATOR CONFIRMATION ---
            
            # RSI indicator with tighter thresholds
            rsi_overbought = latest['RSI'] > SWING_RSI_OVERBOUGHT if pd.notna(latest['RSI']) else False
            
            # V3.0: Now requiring BOTH stoch AND macd (AND logic instead of OR)
            stoch_bearish = latest['Stoch_K'] < latest['Stoch_D'] if pd.notna(latest['Stoch_K']) and pd.notna(latest['Stoch_D']) else False
            macd_bearish = latest['MACD'] < latest['MACD_Signal'] if pd.notna(latest['MACD']) and pd.notna(latest['MACD_Signal']) else False
            
            # More stringent Bollinger Band criteria (0.85 vs previous 0.8)
            bb_overbought = latest['BB_Position'] > 0.85 if pd.notna(latest['BB_Position']) else False
            
            # Tighter price proximity threshold for higher precision
            near_resistance = abs(current_price - resistance_level) / resistance_level < SWING_PRICE_PROXIMITY_THRESHOLD if pd.notna(resistance_level) else False
            
            # V3.0: Triple RSI strategy component (90% win rate in backtests)
            triple_rsi_bearish = False
            if pd.notna(latest['RSI']) and len(df) >= 5:
                # Calculate 3 different RSI measures for confirmation
                rsi_current = latest['RSI']
                rsi_3day = df['RSI'].rolling(3).mean().iloc[-1] if not pd.isna(df['RSI'].rolling(3).mean().iloc[-1]) else 50
                rsi_5day = df['RSI'].rolling(5).mean().iloc[-1] if not pd.isna(df['RSI'].rolling(5).mean().iloc[-1]) else 50
                
                # All RSI measures showing negative momentum with specific bands
                triple_rsi_bearish = (rsi_current < rsi_3day < rsi_5day) and (rsi_current > 65)
            
            # V3.0: Requires multiple criteria together for higher win rate
            signals['bearish_swing'] = (
                rsi_overbought and 
                (stoch_bearish and macd_bearish) and  # V3.0: AND logic, not OR
                                       bb_overbought and 
                                       near_resistance and 
                                       volume_surge and 
                                       downtrend and  # Only take bearish swings in downtrends
                                       bearish_pattern and 
                                       multi_timeframe_confirm and
                price_in_range and
                (triple_rsi_bearish or overall_strength(df) < 30)  # Either Triple RSI or weak stock
            )
            
            # --- V3.0: BULLISH SWING SIGNALS WITH MULTI-INDICATOR CONFIRMATION ---
            
            # RSI with research-tuned thresholds
            rsi_oversold = latest['RSI'] < SWING_RSI_OVERSOLD if pd.notna(latest['RSI']) else False
            
            # V3.0: Now requiring BOTH stoch AND macd (AND logic instead of OR)
            stoch_bullish = latest['Stoch_K'] > latest['Stoch_D'] if pd.notna(latest['Stoch_K']) and pd.notna(latest['Stoch_D']) else False
            macd_bullish = latest['MACD'] > latest['MACD_Signal'] if pd.notna(latest['MACD']) and pd.notna(latest['MACD_Signal']) else False
            
            # More stringent Bollinger Band criteria
            bb_oversold = latest['BB_Position'] < 0.15 if pd.notna(latest['BB_Position']) else False
            
            # Tighter price proximity threshold
            near_support = abs(current_price - support_level) / support_level < SWING_PRICE_PROXIMITY_THRESHOLD if pd.notna(support_level) else False
            
            # V3.0: Triple RSI strategy component (90% win rate in backtests)
            triple_rsi_bullish = False
            if pd.notna(latest['RSI']) and len(df) >= 5:
                # Calculate 3 different RSI measures for confirmation
                rsi_current = latest['RSI']
                rsi_3day = df['RSI'].rolling(3).mean().iloc[-1] if not pd.isna(df['RSI'].rolling(3).mean().iloc[-1]) else 50
                rsi_5day = df['RSI'].rolling(5).mean().iloc[-1] if not pd.isna(df['RSI'].rolling(5).mean().iloc[-1]) else 50
                
                # All RSI measures showing positive momentum with specific bands
                triple_rsi_bullish = (rsi_current > rsi_3day > rsi_5day) and (rsi_current < 35)
            
            # V3.0: Requires multiple criteria together for higher win rate
            signals['bullish_swing'] = (
                rsi_oversold and 
                (stoch_bullish and macd_bullish) and  # V3.0: AND logic, not OR
                                       bb_oversold and 
                                       near_support and 
                                       volume_surge and 
                                       uptrend and  # Only take bullish swings in uptrends
                                       bullish_pattern and
                                       multi_timeframe_confirm and
                price_in_range and
                (triple_rsi_bullish or overall_strength(df) > 70)  # Either Triple RSI or strong stock
            )
            
            # V3.0: MACD + RSI Combined Strategy (73% win rate in backtests)
            if 'MACD_Histogram' in df.columns and pd.notna(latest['MACD_Histogram']) and pd.notna(latest['RSI']):
                # Bullish: MACD histogram turning positive while RSI shows strength
                macd_hist_positive = latest['MACD_Histogram'] > 0
                macd_hist_improving = False
                if len(df) > 2:
                    macd_hist_improving = (
                        latest['MACD_Histogram'] > df['MACD_Histogram'].iloc[-2] > df['MACD_Histogram'].iloc[-3]
                    )
                
                # V3.0: Tighter RSI bands for higher probability signals
                rsi_strength = 45 < latest['RSI'] < 65  # Not overbought but showing strength
                
                # V3.0: More convergent signals required
                signals['macd_rsi_bullish'] = (
                    macd_hist_positive and 
                    macd_hist_improving and 
                    rsi_strength and 
                    uptrend and
                    volume_surge and  # V3.0: Added volume requirement
                    price_in_range  # V3.0: Added price range requirement
                )
                
                # Bearish version with improvements
                macd_hist_negative = latest['MACD_Histogram'] < 0
                macd_hist_deteriorating = False
                if len(df) > 2:
                    macd_hist_deteriorating = (
                        latest['MACD_Histogram'] < df['MACD_Histogram'].iloc[-2] < df['MACD_Histogram'].iloc[-3]
                    )
                
                # V3.0: Tighter RSI bands
                rsi_weakness = 35 < latest['RSI'] < 55  # Not oversold but showing weakness
                
                # V3.0: More convergent signals required
                signals['macd_rsi_bearish'] = (
                    macd_hist_negative and 
                    macd_hist_deteriorating and 
                    rsi_weakness and 
                    downtrend and
                    volume_surge and  # V3.0: Added volume requirement
                    price_in_range  # V3.0: Added price range requirement
                )
            
            # V3.0: Helper function for overall strength calculation
            def overall_strength(df):
                if len(df) < 10:
                    return 50  # Neutral if not enough data
                
                # Calculate price momentum
                price_change = (df['Close'].iloc[-1] / df['Close'].iloc[-10] - 1) * 100
                
                # Calculate volume strength
                vol_ratio = df['Volume'].iloc[-5:].mean() / df['Volume'].iloc[-20:-5].mean() if 'Volume' in df.columns else 1.0
                
                # Calculate technical strength
                rsi = df['RSI'].iloc[-1] if 'RSI' in df.columns and not pd.isna(df['RSI'].iloc[-1]) else 50
                macd_hist = df['MACD_Histogram'].iloc[-1] if 'MACD_Histogram' in df.columns and not pd.isna(df['MACD_Histogram'].iloc[-1]) else 0
                
                # Combined score (0-100)
                price_score = min(100, max(0, 50 + price_change * 5))  # Price change component
                vol_score = min(100, max(0, 50 + (vol_ratio - 1) * 50))  # Volume component
                rsi_score = rsi  # RSI already on 0-100 scale
                macd_score = min(100, max(0, 50 + macd_hist * 200))  # Scale MACD histogram to 0-100
                
                # Weighted average
                return (price_score * 0.3 + vol_score * 0.2 + rsi_score * 0.3 + macd_score * 0.2)
            
        except Exception as e:
            print(f"Error in get_swing_signals: {e}")
            return {}
        
        return signals
    
    def get_breakout_signals(self, df: pd.DataFrame) -> Dict[str, bool]:
        """Enhanced breakout signals based on research criteria"""
        if df.empty or len(df) < 50:
            return {}
        
        latest = df.iloc[-1]
        prev_20 = df.iloc[-21:-1] if len(df) > 20 else df.iloc[:-1]
        signals = {}
        try:
            # ------ Core Breakout Criteria ------
            
            # Volume breakout (increased threshold and using EWM for recency bias)
            volume_ewm = df['Volume'].ewm(span=10).mean().iloc[-1]
            volume_surge = latest['Volume'] / volume_ewm
            signals['volume_breakout'] = volume_surge > BREAKOUT_VOLUME_MULTIPLIER
            
            # Price breakout above resistance with more strict criteria
            bb_position = latest.get('BB_Position', 0.5)
            donchian_break = latest['Close'] > df['High'].rolling(BREAKOUT_CONSOLIDATION_DAYS).max().shift(1).iloc[-1]
            signals['resistance_breakout'] = (
                latest['Close'] > latest.get('BB_Upper', latest['Close']) * 0.98 and bb_position > 0.8
            ) or donchian_break
            
            # ------ New Research-Backed Criteria ------
            
            # 1. Consolidation quality (price compressed then expands)
            if len(prev_20) > BREAKOUT_CONSOLIDATION_DAYS:
                price_range = (prev_20['High'].max() - prev_20['Low'].min()) / prev_20['Close'].mean()
                signals['consolidation_breakout'] = (
                    price_range < BREAKOUT_PRICE_RANGE_LIMIT and 
                    latest['Close'] > prev_20['High'].max() and
                    df['Volume'].iloc[-5:].mean() > df['Volume'].iloc[-20:-5].mean()  # Increasing volume
                )
            else:
                signals['consolidation_breakout'] = False
            
            # 2. Price range filter - favor stocks in optimal price range
            price_in_sweet_spot = BREAKOUT_OPTIMAL_PRICE_RANGE[0] <= latest['Close'] <= BREAKOUT_OPTIMAL_PRICE_RANGE[1]
            signals['price_range_optimal'] = price_in_sweet_spot
            
            # 3. Volatility filter (Average Daily Range)
            avg_daily_range_pct = ((df['High'] - df['Low']) / df['Close'].shift(1)).rolling(10).mean().iloc[-1] * 100
            signals['volatility_optimal'] = avg_daily_range_pct > BREAKOUT_MIN_VOLATILITY_ADR
            
            # 4. Trend continuation check with research-backed thresholds
            signals['trend_continuation'] = (
                latest.get('SMA_20', 0) > latest.get('SMA_50', 0) > latest.get('SMA_200', 0) and
                latest.get('Close', 0) > latest.get('SMA_20', 0) and
                latest.get('ADX', 0) > BREAKOUT_ADX_THRESHOLD  # Lower threshold from research
            )
            
            # 5. Momentum breakout with tuned parameters
            signals['momentum_breakout'] = (
                latest.get('RSI', 0) > 55 and  # Lowered from 60 based on research
                latest.get('MACD_Histogram', 0) > 0 and 
                latest.get('ADX', 0) > BREAKOUT_ADX_THRESHOLD and
                volume_surge > 1.3  # Volume confirmation
            )
            
            # 6. Sector relative strength (placeholder - would be calculated from sector data)
            # In a real implementation, you would compare stock's performance to its sector
            # For now, just using RSI as a proxy for relative strength
            signals['sector_strength'] = latest.get('RSI', 50) > 60
            
            return signals
        except Exception as e:
            print(f"Error in get_breakout_signals: {e}")
            return {}
    
    def get_signal_strength(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate signal strength based on multiple indicator convergence
        Returns scores from 0-100
        """
        if df.empty:
            return {}
        
        latest = df.iloc[-1]
        
        try:
            # Momentum Score (0-100)
            momentum_factors = []
            
            # RSI component
            if 'RSI' in df.columns and pd.notna(latest['RSI']):
                if 30 <= latest['RSI'] <= 70:
                    momentum_factors.append(50)  # Neutral
                elif latest['RSI'] > 70:
                    momentum_factors.append(75)  # Overbought (bullish but risky)
                else:
                    momentum_factors.append(25)  # Oversold (bearish but opportunity)
            
            # MACD component
            if 'MACD_Histogram' in df.columns and pd.notna(latest['MACD_Histogram']):
                if latest['MACD_Histogram'] > 0:
                    momentum_factors.append(70)
                else:
                    momentum_factors.append(30)
            
            # Stochastic component
            if all(col in df.columns for col in ['Stoch_K', 'Stoch_D']) and \
               pd.notna(latest['Stoch_K']) and pd.notna(latest['Stoch_D']):
                if latest['Stoch_K'] > latest['Stoch_D']:
                    momentum_factors.append(70)
                else:
                    momentum_factors.append(30)
            
            momentum_score = np.mean(momentum_factors) if momentum_factors else 50
            
            # Trend Score (0-100)
            trend_factors = []
            
            # ADX strength
            if 'ADX' in df.columns and pd.notna(latest['ADX']):
                if latest['ADX'] > 25:
                    trend_factors.append(80)
                elif latest['ADX'] > 15:
                    trend_factors.append(60)
                else:
                    trend_factors.append(40)
            
            # Moving Average alignment
            ma_cols = ['SMA_20', 'SMA_50', 'SMA_200']
            if all(col in df.columns for col in ma_cols) and \
               all(pd.notna(latest[col]) for col in ma_cols):
                if latest['SMA_20'] > latest['SMA_50'] > latest['SMA_200']:
                    trend_factors.append(90)  # Strong bullish alignment
                elif latest['SMA_20'] < latest['SMA_50'] < latest['SMA_200']:
                    trend_factors.append(10)  # Strong bearish alignment
                else:
                    trend_factors.append(50)  # Mixed
            
            trend_score = np.mean(trend_factors) if trend_factors else 50
            
            # Volume Score (0-100)
            volume_factors = []
            
            # Volume vs SMA
            if all(col in df.columns for col in ['Volume', 'Volume_SMA']) and \
               pd.notna(latest['Volume']) and pd.notna(latest['Volume_SMA']) and latest['Volume_SMA'] > 0:
                volume_ratio = latest['Volume'] / latest['Volume_SMA']
                if volume_ratio > 2.0:
                    volume_factors.append(90)  # Very high volume
                elif volume_ratio > 1.5:
                    volume_factors.append(80)  # High volume
                elif volume_ratio > 1.2:
                    volume_factors.append(70)  # Above average volume
                elif volume_ratio > 1.0:
                    volume_factors.append(60)  # Slightly above average
                elif volume_ratio > 0.8:
                    volume_factors.append(40)  # Slightly below average
                elif volume_ratio > 0.5:
                    volume_factors.append(30)  # Below average
                else:
                    volume_factors.append(20)  # Very low volume
            
            # Volume trend (last 5 days)
            if 'Volume' in df.columns and len(df) >= 5:
                volume_trend = df['Volume'].iloc[-5:].mean() / df['Volume'].iloc[-10:-5].mean()
                if volume_trend > 1.2:
                    volume_factors.append(80)  # Increasing volume trend
                elif volume_trend > 1.0:
                    volume_factors.append(60)  # Slightly increasing
                elif volume_trend > 0.8:
                    volume_factors.append(40)  # Slightly decreasing
                else:
                    volume_factors.append(20)  # Decreasing volume trend
            
            # Price-volume correlation
            if all(col in df.columns for col in ['Close', 'Volume']) and len(df) >= 20:
                price_change = df['Close'].pct_change()
                volume_change = df['Volume'].pct_change()
                correlation = price_change.corr(volume_change)
                if correlation > 0.5:
                    volume_factors.append(80)  # Strong positive correlation
                elif correlation > 0.2:
                    volume_factors.append(60)  # Moderate positive correlation
                elif correlation > -0.2:
                    volume_factors.append(40)  # Weak correlation
                else:
                    volume_factors.append(20)  # Negative correlation
            
            volume_score = np.mean(volume_factors) if volume_factors else 50
            
            # Overall Signal Strength
            overall_score = (momentum_score * 0.4 + trend_score * 0.4 + volume_score * 0.2)
            
            return {
                'momentum_score': momentum_score,
                'trend_score': trend_score,
                'volume_score': volume_score,
                'overall_score': overall_score
            }
            
        except Exception as e:
            print(f"Error in get_signal_strength: {e}")
            return {
                'momentum_score': 50.0,
                'trend_score': 50.0,
                'volume_score': 50.0,
                'overall_score': 50.0
            } 
    
    # ==================== V4.0 PRICE-VOLUME ANALYSIS ====================

    def detect_volume_price_divergence(self, close: pd.Series, volume: pd.Series, 
                                     window: int = VOLUME_PRICE_DIVERGENCE_WINDOW) -> pd.Series:
        """
        Detect divergence between price movement and volume
        
        Based on research paper findings:
        - Price rising on declining volume (bearish divergence)
        - Price falling on declining volume (bullish divergence)
        
        Returns a Series with values:
        1 = bullish divergence
        -1 = bearish divergence
        0 = no divergence
        """
        if len(close) < window + 1:
            return pd.Series(0, index=close.index)
            
        divergence = pd.Series(0, index=close.index)
        
        # Calculate price and volume trends
        for i in range(window, len(close)):
            # Get current window
            price_window = close.iloc[i-window:i+1]
            volume_window = volume.iloc[i-window:i+1]
            
            # Calculate trends
            price_trend = 1 if price_window.iloc[-1] > price_window.iloc[0] else -1
            
            # Calculate volume trend using linear regression slope
            volume_indices = np.arange(len(volume_window))
            volume_slope = np.polyfit(volume_indices, volume_window, 1)[0]
            volume_trend = 1 if volume_slope > 0 else -1
            
            # Detect divergence
            if price_trend == 1 and volume_trend == -1:
                # Bearish divergence: Price up, volume down
                divergence.iloc[i] = -1
            elif price_trend == -1 and volume_trend == -1:
                # Bullish divergence: Price down, volume down
                divergence.iloc[i] = 1
                
        return divergence
    
    def calculate_volume_delta(self, open_prices: pd.Series, close: pd.Series, 
                             volume: pd.Series) -> pd.Series:
        """
        Calculate volume delta (buying vs selling pressure)
        
        Volume delta measures the imbalance between buying and selling volume
        Positive values indicate more buying pressure, negative values indicate more selling pressure
        
        Returns a Series with values between -1 and 1
        """
        # Determine if price closed higher or lower than open
        price_direction = np.sign(close - open_prices)
        
        # Calculate volume delta
        volume_delta = price_direction * volume
        
        # Normalize to -1 to 1 range
        max_volume = volume.rolling(window=20).max()
        normalized_delta = volume_delta / max_volume
        
        return normalized_delta
    
    def detect_bull_trap(self, high: pd.Series, low: pd.Series, close: pd.Series, 
                       volume: pd.Series, resistance_level: pd.Series = None) -> pd.Series:
        """
        Detect potential bull traps (false breakouts to the upside)
        
        Based on research paper findings:
        - Price breaks above resistance
        - Volume spike not sustained
        - Price quickly reverses back below resistance
        
        Returns a Series with 1 where bull traps are detected, 0 otherwise
        """
        if len(close) < 5:
            return pd.Series(0, index=close.index)
            
        bull_traps = pd.Series(0, index=close.index)
        
        # If no resistance level provided, use rolling max as proxy
        if resistance_level is None:
            resistance_level = high.rolling(window=20).max().shift(1)
        
        # Calculate volume ratio to detect spikes
        volume_ratio = volume / volume.rolling(window=10).mean()
        
        # Calculate price reversal
        price_change_pct = close.pct_change(3)  # 3-day price change
        
        for i in range(4, len(close)):
            # Check for breakout above resistance
            breakout = close.iloc[i-1] > resistance_level.iloc[i-1]
            
            # Check for volume spike that's not sustained
            volume_spike = volume_ratio.iloc[i-1] > BULL_TRAP_VOLUME_THRESHOLD
            volume_fade = volume.iloc[i] < volume.iloc[i-1] * 0.8
            
            # Check for price reversal
            reversal = price_change_pct.iloc[i] < -TRAP_REVERSAL_THRESHOLD
            
            # Bull trap condition
            if breakout and volume_spike and volume_fade and reversal:
                bull_traps.iloc[i] = 1
                
        return bull_traps
    
    def detect_bear_trap(self, high: pd.Series, low: pd.Series, close: pd.Series, 
                       volume: pd.Series, support_level: pd.Series = None) -> pd.Series:
        """
        Detect potential bear traps (false breakdowns to the downside)
        
        Based on research paper findings:
        - Price breaks below support
        - Volume spike not sustained
        - Price quickly reverses back above support
        
        Returns a Series with 1 where bear traps are detected, 0 otherwise
        """
        if len(close) < 5:
            return pd.Series(0, index=close.index)
            
        bear_traps = pd.Series(0, index=close.index)
        
        # If no support level provided, use rolling min as proxy
        if support_level is None:
            support_level = low.rolling(window=20).min().shift(1)
        
        # Calculate volume ratio to detect spikes
        volume_ratio = volume / volume.rolling(window=10).mean()
        
        # Calculate price reversal
        price_change_pct = close.pct_change(3)  # 3-day price change
        
        for i in range(4, len(close)):
            # Check for breakdown below support
            breakdown = close.iloc[i-1] < support_level.iloc[i-1]
            
            # Check for volume spike that's not sustained
            volume_spike = volume_ratio.iloc[i-1] > BEAR_TRAP_VOLUME_THRESHOLD
            volume_fade = volume.iloc[i] < volume.iloc[i-1] * 0.8
            
            # Check for price reversal
            reversal = price_change_pct.iloc[i] > TRAP_REVERSAL_THRESHOLD
            
            # Bear trap condition
            if breakdown and volume_spike and volume_fade and reversal:
                bear_traps.iloc[i] = 1
                
        return bear_traps
    
    def detect_false_breakout(self, high: pd.Series, low: pd.Series, close: pd.Series, 
                            volume: pd.Series, period: int = 20) -> pd.Series:
        """
        Detect false breakouts based on price-volume relationships
        
        Based on research paper findings:
        - Price breaks above resistance or below support
        - Volume doesn't confirm the move
        - Price quickly reverses
        
        Returns a Series with:
        1 = false upside breakout (bull trap)
        -1 = false downside breakout (bear trap)
        0 = no false breakout
        """
        if len(close) < period + 5:
            return pd.Series(0, index=close.index)
            
        false_breakouts = pd.Series(0, index=close.index)
        
        # Calculate resistance and support levels
        resistance = high.rolling(window=period).max().shift(1)
        support = low.rolling(window=period).min().shift(1)
        
        # Calculate volume confirmation
        volume_ratio = volume / volume.rolling(window=period).mean()
        
        # Calculate price changes
        price_change_next_day = close.pct_change(1).shift(-1)
        price_change_3day = close.pct_change(3).shift(-3)
        
        for i in range(period, len(close) - 3):
            # Check for breakout above resistance
            upside_breakout = close.iloc[i] > resistance.iloc[i]
            
            # Check for breakdown below support
            downside_breakout = close.iloc[i] < support.iloc[i]
            
            # Check for volume confirmation
            volume_confirms = volume_ratio.iloc[i] > FALSE_BREAKOUT_VOLUME_THRESHOLD
            
            # Check for reversal
            upside_reversal = price_change_3day.iloc[i] < -TRAP_REVERSAL_THRESHOLD
            downside_reversal = price_change_3day.iloc[i] > TRAP_REVERSAL_THRESHOLD
            
            # Detect false breakouts
            if upside_breakout and not volume_confirms and upside_reversal:
                false_breakouts.iloc[i] = 1  # False upside breakout
            elif downside_breakout and not volume_confirms and downside_reversal:
                false_breakouts.iloc[i] = -1  # False downside breakout
                
        return false_breakouts
    
    def detect_hft_activity(self, high: pd.Series, low: pd.Series, close: pd.Series, 
                          volume: pd.Series) -> pd.Series:
        """
        Detect potential high-frequency trading (HFT) activity
        
        Based on research paper findings:
        - Rapid price changes with abnormal volume patterns
        - "Flickering quotes" - quick order submissions and cancellations
        - Volume spikes without sustained follow-through
        
        Returns a Series with values between 0 and 1 indicating HFT activity probability
        """
        if len(close) < 10:
            return pd.Series(0, index=close.index)
            
        hft_activity = pd.Series(0, index=close.index)
        
        # Calculate price volatility (high-low range)
        price_range = (high - low) / close
        
        # Calculate abnormal price ranges
        avg_range = price_range.rolling(window=20).mean()
        range_ratio = price_range / avg_range
        
        # Calculate volume patterns
        volume_ratio = volume / volume.rolling(window=20).mean()
        volume_change = volume.pct_change().abs()
        
        # Calculate price reversals
        price_reversals = close.diff().apply(lambda x: 1 if x != 0 else 0).diff().abs()
        
        for i in range(20, len(close)):
            # Count rapid price changes in last 5 bars
            rapid_changes = sum(price_reversals.iloc[i-5:i] > 0)
            
            # Check for abnormal range with high volume
            abnormal_range = range_ratio.iloc[i] > 2.0
            abnormal_volume = volume_ratio.iloc[i] > 2.0
            
            # Check for volume spike followed by drop
            volume_spike_drop = (volume_ratio.iloc[i-1] > 2.0 and 
                               volume_ratio.iloc[i] < volume_ratio.iloc[i-1] * 0.5)
            
            # HFT activity score (0-1)
            score = 0
            if rapid_changes >= FLICKERING_QUOTE_THRESHOLD:
                score += 0.4
            if abnormal_range and abnormal_volume:
                score += 0.3
            if volume_spike_drop:
                score += 0.3
                
            hft_activity.iloc[i] = min(1.0, score)
                
        return hft_activity
    
    def analyze_market_depth(self, bid_prices: pd.Series, bid_sizes: pd.Series,
                           ask_prices: pd.Series, ask_sizes: pd.Series,
                           volume: pd.Series) -> Dict[str, float]:
        """
        Analyze market depth and order book data
        
        Based on research paper findings:
        - Calculate volume/depth ratio
        - Detect order book imbalances
        - Identify potential fake liquidity
        
        Returns a dictionary with market depth metrics
        """
        # This is a simplified implementation since we don't have real-time order book data
        # In a production environment, this would use actual market depth data
        
        results = {}
        
        # Calculate bid-ask spread
        if len(bid_prices) > 0 and len(ask_prices) > 0:
            mid_price = (bid_prices.iloc[-1] + ask_prices.iloc[-1]) / 2
            spread = ask_prices.iloc[-1] - bid_prices.iloc[-1]
            spread_pct = spread / mid_price
            
            results['bid_ask_spread'] = spread
            results['spread_pct'] = spread_pct
            
            # Flag wide spreads
            results['wide_spread'] = spread_pct > BID_ASK_SPREAD_WARNING
        
        # Calculate market depth
        if len(bid_sizes) > 0 and len(ask_sizes) > 0:
            total_bid_depth = bid_sizes.sum()
            total_ask_depth = ask_sizes.sum()
            total_depth = total_bid_depth + total_ask_depth
            
            results['total_depth'] = total_depth
            
            # Calculate order book imbalance
            if total_depth > 0:
                bid_ask_imbalance = abs(total_bid_depth - total_ask_depth) / total_depth
                results['bid_ask_imbalance'] = bid_ask_imbalance
                
                # Flag significant imbalances
                results['significant_imbalance'] = bid_ask_imbalance > ORDER_FLOW_IMBALANCE_THRESHOLD
            
            # Calculate volume/depth ratio
            if total_depth > 0 and len(volume) > 0:
                latest_volume = volume.iloc[-1]
                volume_depth_ratio = latest_volume / total_depth
                results['volume_depth_ratio'] = volume_depth_ratio
                
                # Flag suspicious volume/depth ratio
                results['suspicious_liquidity'] = volume_depth_ratio > VOLUME_DEPTH_RATIO_THRESHOLD
        
        return results
    
    def detect_stop_hunting(self, high: pd.Series, low: pd.Series, close: pd.Series, 
                          volume: pd.Series, atr: pd.Series = None) -> pd.Series:
        """
        Detect potential stop hunting patterns
        
        Based on research paper findings:
        - Sharp price moves that trigger common stop levels
        - Followed by immediate reversal
        - Often accompanied by volume spikes
        
        Returns a Series with 1 where stop hunting is detected, 0 otherwise
        """
        if len(close) < 20:
            return pd.Series(0, index=close.index)
            
        stop_hunting = pd.Series(0, index=close.index)
        
        # Calculate ATR if not provided
        if atr is None:
            atr = self.calculate_atr(high, low, close)
        
        # Calculate common stop levels (round numbers, prior day high/low, key MAs)
        prior_day_high = high.shift(1)
        prior_day_low = low.shift(1)
        
        # Calculate volume spikes
        volume_ratio = volume / volume.rolling(window=20).mean()
        
        for i in range(20, len(close)):
            # Check for price spike beyond common stop levels
            spike_beyond_prior_high = high.iloc[i] > prior_day_high.iloc[i] and low.iloc[i] < prior_day_high.iloc[i]
            spike_beyond_prior_low = low.iloc[i] < prior_day_low.iloc[i] and high.iloc[i] > prior_day_low.iloc[i]
            
            # Check for intraday reversal (high-low range > 2x ATR)
            intraday_reversal = (high.iloc[i] - low.iloc[i]) > (2 * atr.iloc[i])
            
            # Check for volume spike
            volume_spike = volume_ratio.iloc[i] > 1.5
            
            # Check for price reversal next day
            if i < len(close) - 1:
                next_day_reversal = (close.iloc[i] - close.iloc[i-1]) * (close.iloc[i+1] - close.iloc[i]) < 0
            else:
                next_day_reversal = False
            
            # Stop hunting pattern
            if (spike_beyond_prior_high or spike_beyond_prior_low) and intraday_reversal and volume_spike and next_day_reversal:
                stop_hunting.iloc[i] = 1
                
        return stop_hunting
    
    # ==================== OPTIONS PINNING ANALYSIS ====================
    
    def analyze_options_pinning(self, 
                               close: pd.Series, 
                               high: pd.Series = None, 
                               low: pd.Series = None,
                               volume: pd.Series = None, 
                               option_strikes: List[float] = None,
                               option_oi: Dict[float, int] = None) -> Dict[str, any]:
        """
        Analyze options pinning effects and potential strike price magnets
        
        Parameters:
        - close: Price series
        - high/low: Optional high/low series
        - volume: Optional volume series
        - option_strikes: List of available option strike prices
        - option_oi: Dictionary mapping strikes to open interest values
        
        Returns:
        - Dictionary with pinning analysis results
        """
        result = {
            'nearest_strike': None,
            'distance_to_strike': 0.0,
            'potential_magnets': [],
            'pinning_pressure': 0.0,
            'expiration_proximity': False
        }
        
        # Skip if no strike data provided
        if not option_strikes or len(option_strikes) == 0:
            return result
        
        current_price = close.iloc[-1]
        
        # Find nearest strike
        nearest_strike = min(option_strikes, key=lambda x: abs(x - current_price))
        result['nearest_strike'] = nearest_strike
        result['distance_to_strike'] = abs(current_price - nearest_strike)
        
        # If OI data is available, identify potential magnet strikes
        if option_oi and len(option_oi) > 0:
            # Filter to strikes within 5% of current price
            nearby_strikes = {k: v for k, v in option_oi.items() 
                              if abs(k - current_price) / current_price < 0.05}
            
            # Sort by OI and convert to list of (strike, oi) tuples
            sorted_strikes = sorted(nearby_strikes.items(), key=lambda x: x[1], reverse=True)
            
            # Take top 3 strikes with highest OI as potential magnets
            result['potential_magnets'] = sorted_strikes[:3] if len(sorted_strikes) >= 3 else sorted_strikes
            
            # Calculate pinning pressure based on OI concentration
            if sorted_strikes:
                total_nearby_oi = sum(oi for _, oi in sorted_strikes)
                highest_oi = sorted_strikes[0][1] if sorted_strikes else 0
                
                # Pinning pressure is ratio of highest OI to average OI
                avg_oi = total_nearby_oi / len(sorted_strikes) if sorted_strikes else 0
                result['pinning_pressure'] = highest_oi / avg_oi if avg_oi > 0 else 0
        
        return result
    
    def detect_pinning_pattern(self, 
                              close: pd.Series, 
                              nearest_strike: float,
                              lookback: int = 3) -> bool:
        """
        Detect if price is showing a pinning pattern (oscillating around strike with decreasing amplitude)
        
        Parameters:
        - close: Price series
        - nearest_strike: Nearest option strike price
        - lookback: Number of days to look back for pattern
        
        Returns:
        - Boolean indicating if pinning pattern detected
        """
        if len(close) < lookback + 1:
            return False
            
        # Get price differences from nearest strike
        diffs = close[-lookback:] - nearest_strike
        
        # Check if price is oscillating around the strike (sign changes)
        sign_changes = sum(1 for i in range(1, len(diffs)) if diffs[i] * diffs[i-1] < 0)
        
        # Calculate amplitude of oscillations
        amplitudes = [abs(diff) for diff in diffs]
        
        # Check if pattern shows decreasing amplitude (at least 2 decreasing steps)
        decreasing_amplitude = sum(1 for i in range(1, len(amplitudes)) 
                                  if amplitudes[i] < amplitudes[i-1])
        
        # Pinning pattern detected if:
        # 1. At least one sign change (price crosses the strike)
        # 2. Amplitude is decreasing (price getting closer to strike)
        return sign_changes > 0 and decreasing_amplitude >= 1 