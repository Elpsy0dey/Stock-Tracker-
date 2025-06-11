"""
Risk Management Module for Trading Portfolio Tracker

Implements risk management strategies based on research study:
- ≤5% portfolio loss rule
- Position sizing based on volatility (ATR)
- Stop-loss calculations
- Portfolio exposure monitoring
- Diversification metrics
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from config.settings import *

class RiskManager:
    """Risk management engine for portfolio and position sizing"""
    
    def __init__(self):
        self.max_portfolio_risk = MAX_PORTFOLIO_RISK
        self.max_position_risk = MAX_POSITION_RISK
        
    def calculate_position_size(self, 
                               portfolio_value: float,
                               entry_price: float,
                               stop_loss_price: float,
                               risk_per_trade: float = None) -> Dict[str, float]:
        """
        Calculate optimal position size based on risk management rules
        
        Args:
            portfolio_value: Total portfolio value
            entry_price: Planned entry price
            stop_loss_price: Stop loss price
            risk_per_trade: Risk per trade as decimal (default uses MAX_POSITION_RISK)
            
        Returns:
            Dictionary with position sizing details
        """
        if risk_per_trade is None:
            risk_per_trade = self.max_position_risk
        
        # Calculate risk per share
        risk_per_share = abs(entry_price - stop_loss_price)
        
        if risk_per_share == 0:
            return {
                'shares': 0,
                'position_value': 0,
                'risk_amount': 0,
                'risk_percentage': 0,
                'error': 'Invalid stop loss - same as entry price'
            }
        
        # Calculate maximum risk amount
        max_risk_amount = portfolio_value * risk_per_trade
        
        # Calculate number of shares
        shares = int(max_risk_amount / risk_per_share)
        
        # Calculate actual position value and risk
        position_value = shares * entry_price
        actual_risk_amount = shares * risk_per_share
        actual_risk_percentage = (actual_risk_amount / portfolio_value) * 100
        
        # Position size as percentage of portfolio
        position_percentage = (position_value / portfolio_value) * 100
        
        return {
            'shares': shares,
            'position_value': position_value,
            'position_percentage': position_percentage,
            'risk_amount': actual_risk_amount,
            'risk_percentage': actual_risk_percentage,
            'risk_per_share': risk_per_share,
            'max_risk_amount': max_risk_amount,
            'shares_at_max_risk': int(max_risk_amount / risk_per_share)
        }
    
    def calculate_atr_based_stop_loss(self, 
                                    current_price: float, 
                                    atr: float, 
                                    multiplier: float = 2.0,
                                    trade_type: str = 'swing') -> Dict[str, float]:
        """
        Calculate ATR-based stop loss levels
        
        Args:
            current_price: Current stock price
            atr: Average True Range value
            multiplier: ATR multiplier (default 2.0)
            trade_type: 'swing' or 'breakout' (affects multiplier)
            
        Returns:
            Dictionary with stop loss calculations
        """
        # Adjust multiplier based on trade type
        if trade_type == 'swing':
            default_multiplier = 2.0
        elif trade_type == 'breakout':
            default_multiplier = 3.0
        else:
            default_multiplier = 2.0
        
        if multiplier == 2.0:  # If using default, apply trade-type adjustment
            multiplier = default_multiplier
        
        # Calculate stop loss levels
        long_stop_loss = current_price - (atr * multiplier)
        short_stop_loss = current_price + (atr * multiplier)
        
        # Calculate risk percentages
        long_risk_pct = ((current_price - long_stop_loss) / current_price) * 100
        short_risk_pct = ((short_stop_loss - current_price) / current_price) * 100
        
        return {
            'long_stop_loss': long_stop_loss,
            'short_stop_loss': short_stop_loss,
            'long_risk_percentage': long_risk_pct,
            'short_risk_percentage': short_risk_pct,
            'atr_multiplier': multiplier,
            'atr_value': atr
        }
    
    def calculate_volatility_adjusted_position_size(self,
                                                  portfolio_value: float,
                                                  entry_price: float,
                                                  atr: float,
                                                  target_risk: float = None) -> Dict[str, float]:
        """
        Calculate position size adjusted for volatility using ATR
        
        Args:
            portfolio_value: Total portfolio value
            entry_price: Planned entry price
            atr: Average True Range value
            target_risk: Target risk as decimal (default uses MAX_POSITION_RISK)
            
        Returns:
            Dictionary with volatility-adjusted position sizing
        """
        if target_risk is None:
            target_risk = self.max_position_risk
        
        # Calculate target dollar risk
        target_dollar_risk = portfolio_value * target_risk
        
        # Calculate position size based on ATR risk
        # Using 2x ATR as risk per share (common practice)
        risk_per_share = atr * 2.0
        
        if risk_per_share == 0:
            return {
                'shares': 0,
                'position_value': 0,
                'volatility_adjusted': False,
                'error': 'ATR is zero'
            }
        
        # Calculate shares
        shares = int(target_dollar_risk / risk_per_share)
        position_value = shares * entry_price
        
        # Calculate actual risk metrics
        actual_risk_amount = shares * risk_per_share
        actual_risk_percentage = (actual_risk_amount / portfolio_value) * 100
        position_percentage = (position_value / portfolio_value) * 100
        
        # Volatility assessment
        volatility_score = (atr / entry_price) * 100  # ATR as % of price
        
        return {
            'shares': shares,
            'position_value': position_value,
            'position_percentage': position_percentage,
            'risk_amount': actual_risk_amount,
            'risk_percentage': actual_risk_percentage,
            'volatility_score': volatility_score,
            'risk_per_share': risk_per_share,
            'volatility_adjusted': True
        }
    
    def assess_portfolio_risk(self, portfolio_details: Dict, 
                            current_drawdown: float = 0) -> Dict[str, float]:
        """
        Assess overall portfolio risk levels
        
        Args:
            portfolio_details: Dictionary of current positions
            current_drawdown: Current portfolio drawdown percentage
            
        Returns:
            Dictionary with risk assessment metrics
        """
        if not portfolio_details:
            return {'total_risk': 0, 'risk_level': 'Low'}
        
        # Calculate total portfolio value
        total_value = sum(pos['market_value'] for pos in portfolio_details.values())
        
        # Calculate individual position risks (assuming 5% stop loss)
        total_risk_amount = 0
        position_risks = {}
        
        for symbol, position in portfolio_details.items():
            position_value = position['market_value']
            position_percentage = (position_value / total_value) * 100
            
            # Estimate risk (5% of position value as default)
            estimated_risk = position_value * DEFAULT_STOP_LOSS
            total_risk_amount += estimated_risk
            
            position_risks[symbol] = {
                'position_percentage': position_percentage,
                'estimated_risk': estimated_risk,
                'risk_percentage': (estimated_risk / total_value) * 100
            }
        
        # Calculate total portfolio risk percentage
        total_risk_percentage = (total_risk_amount / total_value) * 100
        
        # Risk level assessment
        if total_risk_percentage <= 3:
            risk_level = 'Low'
        elif total_risk_percentage <= 7:
            risk_level = 'Medium'
        else:
            risk_level = 'High'
        
        # Concentration risk (largest position percentage)
        max_position_percentage = max((pos['position_percentage'] for pos in position_risks.values()), default=0)
        
        # Diversification score (simple Herfindahl index)
        position_weights = [pos['position_percentage']/100 for pos in position_risks.values()]
        herfindahl_index = sum(w**2 for w in position_weights)
        diversification_score = (1 - herfindahl_index) * 100
        
        return {
            'total_risk_percentage': total_risk_percentage,
            'total_risk_amount': total_risk_amount,
            'risk_level': risk_level,
            'position_risks': position_risks,
            'max_position_percentage': max_position_percentage,
            'diversification_score': diversification_score,
            'current_drawdown': current_drawdown,
            'positions_count': len(portfolio_details),
            'risk_budget_used': total_risk_percentage / (self.max_portfolio_risk * 100)
        }
    
    def calculate_kelly_criterion(self, win_rate: float, avg_win: float, avg_loss: float) -> float:
        """
        Calculate Kelly Criterion for optimal position sizing
        
        Args:
            win_rate: Win rate as percentage (0-100)
            avg_win: Average winning trade amount
            avg_loss: Average losing trade amount (positive number)
            
        Returns:
            Kelly percentage (as decimal)
        """
        if avg_loss == 0 or win_rate == 0:
            return 0
        
        win_prob = win_rate / 100
        loss_prob = 1 - win_prob
        
        win_loss_ratio = avg_win / avg_loss
        
        # Kelly formula: f = (bp - q) / b
        # where b = win/loss ratio, p = win probability, q = loss probability
        kelly_fraction = (win_loss_ratio * win_prob - loss_prob) / win_loss_ratio
        
        # Cap Kelly at reasonable levels (typically 25% max)
        kelly_fraction = max(0, min(kelly_fraction, 0.25))
        
        return kelly_fraction
    
    def generate_position_sizing_recommendation(self,
                                              portfolio_value: float,
                                              entry_price: float,
                                              stop_loss_price: float,
                                              atr: float,
                                              win_rate: float = 50,
                                              avg_win: float = 100,
                                              avg_loss: float = 100) -> Dict:
        """
        Generate comprehensive position sizing recommendation
        
        Returns multiple position sizing methods for comparison
        """
        recommendations = {}
        
        # 1. Fixed risk method (2% risk)
        fixed_risk = self.calculate_position_size(
            portfolio_value, entry_price, stop_loss_price, self.max_position_risk
        )
        recommendations['fixed_risk'] = fixed_risk
        
        # 2. ATR-based method
        atr_based = self.calculate_volatility_adjusted_position_size(
            portfolio_value, entry_price, atr
        )
        recommendations['atr_based'] = atr_based
        
        # 3. Kelly Criterion method
        kelly_fraction = self.calculate_kelly_criterion(win_rate, avg_win, avg_loss)
        kelly_risk = min(kelly_fraction, self.max_position_risk)  # Cap at max risk
        kelly_sizing = self.calculate_position_size(
            portfolio_value, entry_price, stop_loss_price, kelly_risk
        )
        kelly_sizing['kelly_fraction'] = kelly_fraction
        recommendations['kelly_criterion'] = kelly_sizing
        
        # 4. Conservative method (1% risk)
        conservative = self.calculate_position_size(
            portfolio_value, entry_price, stop_loss_price, 0.01
        )
        recommendations['conservative'] = conservative
        
        # Generate final recommendation based on risk profile
        if kelly_fraction > 0.15:  # High confidence
            recommended_method = 'kelly_criterion'
        elif atr_based['volatility_score'] > 5:  # High volatility
            recommended_method = 'conservative'
        else:
            recommended_method = 'fixed_risk'
        
        recommendations['recommended'] = recommended_method
        recommendations['recommended_sizing'] = recommendations[recommended_method]
        
        return recommendations
    
    def check_risk_limits(self, 
                         portfolio_details: Dict,
                         new_position_value: float,
                         new_position_risk: float) -> Dict[str, bool]:
        """
        Check if adding a new position would violate risk limits
        
        Args:
            portfolio_details: Current portfolio positions
            new_position_value: Value of new position to add
            new_position_risk: Risk amount of new position
            
        Returns:
            Dictionary with limit check results
        """
        if not portfolio_details:
            total_value = new_position_value
            current_risk = 0
        else:
            total_value = sum(pos['market_value'] for pos in portfolio_details.values()) + new_position_value
            current_risk = sum(pos['market_value'] * DEFAULT_STOP_LOSS for pos in portfolio_details.values())
        
        # Calculate new total risk
        new_total_risk = current_risk + new_position_risk
        new_risk_percentage = (new_total_risk / total_value) * 100
        
        # Calculate new position as percentage of portfolio
        new_position_percentage = (new_position_value / total_value) * 100
        
        # Check limits
        portfolio_risk_ok = new_risk_percentage <= (self.max_portfolio_risk * 100)
        position_size_ok = new_position_percentage <= 20  # Max 20% in single position
        total_positions_ok = len(portfolio_details) < 20  # Max 20 positions
        
        return {
            'portfolio_risk_ok': portfolio_risk_ok,
            'position_size_ok': position_size_ok,
            'total_positions_ok': total_positions_ok,
            'all_limits_ok': all([portfolio_risk_ok, position_size_ok, total_positions_ok]),
            'new_portfolio_risk': new_risk_percentage,
            'new_position_percentage': new_position_percentage,
            'risk_budget_remaining': (self.max_portfolio_risk * 100) - new_risk_percentage
        }
    
    def calculate_correlation_risk(self, symbols: List[str], 
                                 period: str = "6mo") -> Dict[str, float]:
        """
        Calculate correlation risk between portfolio positions
        (This would require price data integration)
        """
        # Placeholder for correlation analysis
        # In a full implementation, this would:
        # 1. Get price data for all symbols
        # 2. Calculate correlation matrix
        # 3. Assess concentration risk
        # 4. Return correlation metrics
        
        return {
            'avg_correlation': 0.5,  # Placeholder
            'max_correlation': 0.8,  # Placeholder
            'correlation_risk': 'Medium'  # Placeholder
        }
    
    def generate_risk_report(self, 
                           portfolio_details: Dict,
                           trade_history: pd.DataFrame = None) -> Dict:
        """
        Generate comprehensive risk management report
        
        Args:
            portfolio_details: Current portfolio positions
            trade_history: Historical trade data
            
        Returns:
            Comprehensive risk assessment report
        """
        # Portfolio risk assessment
        portfolio_risk = self.assess_portfolio_risk(portfolio_details)
        
        # Historical performance metrics (if trade history available)
        historical_metrics = {}
        if trade_history is not None and not trade_history.empty:
            # Calculate win rate, avg win/loss, etc.
            # This would integrate with portfolio tracker calculations
            pass
        
        # Generate recommendations
        recommendations = []
        
        if portfolio_risk['total_risk_percentage'] > 7:
            recommendations.append("Portfolio risk exceeds 7% - consider reducing position sizes")
        
        if portfolio_risk['max_position_percentage'] > 15:
            recommendations.append("Largest position exceeds 15% - consider diversifying")
        
        if portfolio_risk['diversification_score'] < 60:
            recommendations.append("Low diversification score - consider adding uncorrelated positions")
        
        if portfolio_risk['positions_count'] < 3:
            recommendations.append("Consider adding more positions for better diversification")
        
        return {
            'portfolio_risk': portfolio_risk,
            'historical_metrics': historical_metrics,
            'recommendations': recommendations,
            'risk_summary': {
                'overall_risk_level': portfolio_risk['risk_level'],
                'risk_budget_used': f"{portfolio_risk['risk_budget_used']*100:.1f}%",
                'diversification': 'Good' if portfolio_risk['diversification_score'] > 70 else 'Needs Improvement'
            }
        }
    
    def suggest_stop_loss_levels(self,
                               entry_price: float,
                               technical_support: float,
                               atr: float,
                               trade_type: str = 'swing') -> Dict[str, float]:
        """
        Suggest multiple stop loss levels based on different methods
        
        Args:
            entry_price: Entry price for the trade
            technical_support: Technical support level
            atr: Average True Range
            trade_type: Type of trade ('swing' or 'breakout')
            
        Returns:
            Dictionary with different stop loss suggestions
        """
        stop_loss_levels = {}
        
        # 1. Percentage-based stop loss
        if trade_type == 'swing':
            percent_stop = entry_price * (1 - SWING_STOP_LOSS)
        else:
            percent_stop = entry_price * (1 - BREAKOUT_STOP_LOSS)
        
        stop_loss_levels['percentage_based'] = percent_stop
        
        # 2. ATR-based stop loss
        atr_stops = self.calculate_atr_based_stop_loss(entry_price, atr, trade_type=trade_type)
        stop_loss_levels['atr_based'] = atr_stops['long_stop_loss']
        
        # 3. Technical level stop loss
        if technical_support > 0:
            technical_stop = technical_support * 0.99  # Just below support
            stop_loss_levels['technical_level'] = technical_stop
        
        # 4. Conservative stop loss (tighter)
        conservative_stop = entry_price * 0.97  # 3% stop
        stop_loss_levels['conservative'] = conservative_stop
        
        # Recommend the most appropriate stop loss
        # Generally use the one that provides reasonable risk while respecting technical levels
        if technical_support > 0:
            recommended = max(stop_loss_levels['technical_level'], stop_loss_levels['percentage_based'])
        else:
            recommended = stop_loss_levels['atr_based']
        
        stop_loss_levels['recommended'] = recommended
        
        return stop_loss_levels 