"""
Portfolio Tracker Module for Trading Portfolio Tracker

Refactored core portfolio tracking functionality from the original TradingTracker class
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Tuple, List, Optional
from utils.data_utils import get_current_prices, format_currency, format_percentage
from config.settings import *

class PortfolioTracker:
    def __init__(self, starting_cash=DEFAULT_STARTING_CASH):
        self.starting_cash = starting_cash
        self.trades_df = pd.DataFrame()
        self.portfolio = {}
        self.realized_pnl = 0.0
        self.total_fees = 0.0
        self.current_prices = {}
        self.trade_history = []  # List to store trade history
        
    def load_trades(self, trades_df: pd.DataFrame) -> bool:
        """Load trades DataFrame and calculate portfolio"""
        try:
            self.trades_df = trades_df
            self._calculate_portfolio_and_pnl()
            return True
        except Exception as e:
            print(f"Error loading trades: {str(e)}")
            return False
    
    def _calculate_portfolio_and_pnl(self):
        """Calculate current portfolio positions and realized P&L"""
        self.portfolio = {}
        self.realized_pnl = 0.0
        self.total_fees = 0.0
        self.trade_history = []  # List to store trade history
        
        # Group trades by ticker
        for ticker in self.trades_df['ticker'].unique():
            ticker_trades = self.trades_df[self.trades_df['ticker'] == ticker].copy()
            # CHANGE: Preserve the original order from Excel instead of sorting by date
            # ticker_trades = ticker_trades.sort_values('trade_date')
            
            position = 0
            avg_cost = 0.0
            total_cost = 0.0
            realized_pnl_ticker = 0.0
            entry_date = None
            entry_price = 0.0
            
            for _, trade in ticker_trades.iterrows():
                units = trade['units']
                price = trade['avg_price']
                fees = trade['fees'] if not pd.isna(trade['fees']) else 0
                side = trade['side'].upper()
                trade_date = trade['trade_date']
                
                self.total_fees += fees
                
                if side == 'BUY':
                    # Add to position
                    total_cost += abs(units) * price + fees
                    position += abs(units)
                    if position > 0:
                        avg_cost = total_cost / position
                        if entry_date is None:
                            entry_date = trade_date
                            entry_price = price
                        
                elif side == 'SELL':
                    # Reduce position
                    units_sold = min(abs(units), position)
                    if position >= units_sold:
                        # Calculate realized P&L for sold units
                        proceeds = units_sold * price - fees
                        cost_basis = units_sold * avg_cost
                        realized_pnl_ticker += proceeds - cost_basis
                        
                        # Calculate trade details
                        hold_time = (trade_date - entry_date).days if entry_date else 0
                        pnl = proceeds - cost_basis
                        pnl_pct = (pnl / cost_basis) * 100 if cost_basis > 0 else 0
                        
                        # Store trade information
                        trade_info = {
                            'symbol': ticker,
                            'entry_date': entry_date,
                            'exit_date': trade_date,
                            'entry_price': entry_price,
                            'exit_price': price,
                            'units': units_sold,
                            'pnl': pnl,
                            'pnl_pct': pnl_pct,
                            'hold_time': hold_time,
                            'fees': fees
                        }
                        self.trade_history.append(trade_info)
                        
                        # Update position
                        position -= units_sold
                        if position > 0:
                            total_cost = position * avg_cost
                        else:
                            total_cost = 0
                            avg_cost = 0
                            entry_date = None
                            entry_price = 0.0
            
            # Store current position
            if position > 0:
                self.portfolio[ticker] = {
                    'shares': position,
                    'avg_cost': avg_cost,
                    'total_cost': total_cost
                }
            
            self.realized_pnl += realized_pnl_ticker
    
    def refresh_current_prices(self):
        """Fetch current market prices for portfolio holdings"""
        if self.portfolio:
            symbols = list(self.portfolio.keys())
            self.current_prices = get_current_prices(symbols)
        else:
            self.current_prices = {}
    
    def calculate_portfolio_value(self) -> Tuple[float, Dict]:
        """Calculate current portfolio market value"""
        if not self.current_prices and self.portfolio:
            self.refresh_current_prices()
        
        total_value = 0.0
        portfolio_details = {}
        
        for ticker, position in self.portfolio.items():
            current_price = self.current_prices.get(ticker, position['avg_cost'])
            market_value = position['shares'] * current_price
            unrealized_pnl = market_value - position['total_cost']
            
            portfolio_details[ticker] = {
                'shares': position['shares'],
                'avg_cost': position['avg_cost'],
                'current_price': current_price,
                'market_value': market_value,
                'cost_basis': position['total_cost'],
                'unrealized_pnl': unrealized_pnl,
                'unrealized_pnl_pct': (unrealized_pnl / position['total_cost']) * 100 if position['total_cost'] > 0 else 0
            }
            
            total_value += market_value
        
        return total_value, portfolio_details
    
    def calculate_cash_balance(self) -> float:
        """Calculate current cash balance"""
        # Calculate total money spent on purchases
        buy_trades = self.trades_df[self.trades_df['side'].str.upper() == 'BUY']
        total_purchases = buy_trades['total_value'].sum() if 'total_value' in buy_trades.columns else (buy_trades['units'] * buy_trades['avg_price'] + buy_trades['fees']).sum()
        
        # Calculate total money received from sales
        sell_trades = self.trades_df[self.trades_df['side'].str.upper() == 'SELL']
        total_sales = sell_trades['total_value'].sum() if 'total_value' in sell_trades.columns else (sell_trades['units'] * sell_trades['avg_price'] - sell_trades['fees']).sum()
        
        # Cash balance = starting cash - purchases + sales
        cash_balance = self.starting_cash - total_purchases + abs(total_sales)
        return cash_balance
    
    def get_summary_stats(self) -> Dict:
        """Get comprehensive summary statistics"""
        portfolio_value, portfolio_details = self.calculate_portfolio_value()
        cash_balance = self.calculate_cash_balance()
        total_account_value = portfolio_value + cash_balance
        
        # Calculate total unrealized P&L
        total_unrealized_pnl = sum([pos['unrealized_pnl'] for pos in portfolio_details.values()])
        
        # Trading statistics
        total_trades = len(self.trades_df)
        buy_trades = len(self.trades_df[self.trades_df['side'].str.upper() == 'BUY'])
        sell_trades = len(self.trades_df[self.trades_df['side'].str.upper() == 'SELL'])
        
        return {
            'starting_cash': self.starting_cash,
            'current_cash': cash_balance,
            'portfolio_value': portfolio_value,
            'total_account_value': total_account_value,
            'realized_pnl': self.realized_pnl,
            'unrealized_pnl': total_unrealized_pnl,
            'total_pnl': self.realized_pnl + total_unrealized_pnl,
            'total_return_pct': ((total_account_value - self.starting_cash) / self.starting_cash) * 100,
            'total_fees': self.total_fees,
            'total_trades': total_trades,
            'buy_trades': buy_trades,
            'sell_trades': sell_trades,
            'portfolio_details': portfolio_details
        }
    
    def calculate_monthly_balance_changes(self) -> pd.DataFrame:
        """Calculate monthly account balance changes - strictly follows step-by-step approach"""
        if self.trades_df.empty:
            return pd.DataFrame()
        
        # Ensure trade_date is datetime
        trades = self.trades_df.copy()
        if not pd.api.types.is_datetime64_any_dtype(trades['trade_date']):
            trades['trade_date'] = pd.to_datetime(trades['trade_date'])
        
        # Add month-year column WITHOUT sorting - preserve original order from Excel
        # CHANGE: Removed sort_values call to preserve original trade order
        trades['month_year'] = trades['trade_date'].dt.to_period('M')
        
        # Make sure we have current prices
        self.refresh_current_prices()
        
        # Initialize tracking variables
        monthly_data = []
        current_cash = self.starting_cash
        previous_month_portfolio_value = 0.0
        portfolio_positions = {}  # {ticker: {'shares': shares, 'avg_cost': cost_per_share}}
        
        # Track all closed positions for realized P&L calculations
        closed_positions_pnl = 0.0
        
        # Process each month
        for month_period in sorted(trades['month_year'].unique()):
            # Get trades for this month, but preserve their original order from Excel
            # CHANGE: Using original index to preserve order instead of sorting by date
            month_trades = trades[trades['month_year'] == month_period]
            month_trades = month_trades.sort_index()  # Preserve original order from Excel
            
            # Reset monthly tracking variables
            month_realized_pnl = 0.0
            month_investments = 0.0
            month_returns = 0.0
            month_fees = 0.0
            
            # Track positions opened/closed this month
            positions_closed = {}  # For P&L tracking
            
            # Get month end date for historical pricing
            month_end = month_period.to_timestamp(how='end')
            
            # Process each trade chronologically
            for _, trade in month_trades.iterrows():
                ticker = trade['ticker']
                side = trade['side'].upper()
                
                # Handle numeric conversions safely
                units = abs(float(trade['units'])) if isinstance(trade['units'], str) else abs(float(trade['units']))
                price = float(trade['avg_price']) if isinstance(trade['avg_price'], str) else float(trade['avg_price'])
                fees = float(trade['fees']) if not pd.isna(trade['fees']) else 0.0
                
                # Calculate trade value (without fees)
                trade_value = units * price
                total_cost = trade_value + fees if side == 'BUY' else trade_value
                
                # Track trade data
                month_fees += fees
                
                if side == 'BUY':
                    # Track investment amount
                    month_investments += (trade_value + fees)
                    
                    # Reduce cash
                    current_cash -= (trade_value + fees)
                    
                    # Create/update position
                    if ticker not in portfolio_positions:
                        portfolio_positions[ticker] = {
                            'shares': units,
                            'avg_cost': price + (fees / units) if units > 0 else price,
                            'total_cost': trade_value + fees
                        }
                    else:
                        # Calculate new average cost
                        current_shares = portfolio_positions[ticker]['shares']
                        current_total_cost = portfolio_positions[ticker]['total_cost']
                        new_total_shares = current_shares + units
                        new_total_cost = current_total_cost + trade_value + fees
                        
                        portfolio_positions[ticker]['shares'] = new_total_shares
                        portfolio_positions[ticker]['total_cost'] = new_total_cost
                        portfolio_positions[ticker]['avg_cost'] = new_total_cost / new_total_shares if new_total_shares > 0 else 0
                else:  # SELL
                    # Only process sell if position exists
                    if ticker in portfolio_positions:
                        position = portfolio_positions[ticker]
                        
                        # Only sell what we actually have
                        available_shares = position['shares']
                        if available_shares > 0:
                            # Limit units to available shares
                            actual_units = min(units, available_shares)
                            actual_trade_value = actual_units * price
                            
                            # Track returns
                            month_returns += (actual_trade_value - fees)
                            
                            # Add to cash
                            current_cash += (actual_trade_value - fees)
                            
                            # Calculate realized P&L for the sold portion
                            sold_shares_cost = position['avg_cost'] * actual_units
                            sold_shares_proceeds = actual_trade_value - fees
                            trade_pnl = sold_shares_proceeds - sold_shares_cost
                            
                            # Add to monthly realized P&L
                            month_realized_pnl += trade_pnl
                            
                            # Update position
                            new_shares = position['shares'] - actual_units
                            
                            if new_shares <= 0:
                                # Position fully closed
                                del portfolio_positions[ticker]
                            else:
                                # Reduce position proportionally
                                remaining_cost = position['avg_cost'] * new_shares
                                portfolio_positions[ticker]['shares'] = new_shares
                                portfolio_positions[ticker]['total_cost'] = remaining_cost
                        else:
                            pass  # No shares available, skip
                    else:
                        pass  # No position exists, skip
            
            # Calculate end-of-month portfolio value
            portfolio_value = 0.0
            unrealized_pnl = 0.0
            
            # Determine if month has ended
            is_current_month = (month_period.to_timestamp(how='end') >= datetime.now())
            
            # For current month, ensure we have fresh prices (do this once per month)
            if is_current_month:
                self.refresh_current_prices()
            
            # Value remaining positions - use month-end prices when available
            for ticker, position in portfolio_positions.items():
                # For month-end valuation, get the last trade price of the month
                ticker_month_trades = month_trades[month_trades['ticker'] == ticker]
                
                # Get best price estimate based on month status
                if is_current_month and ticker in self.current_prices and self.current_prices[ticker] > 0:
                    # For current month, prioritize using latest market price
                    month_end_price = self.current_prices[ticker]
                elif not ticker_month_trades.empty:
                    # Use the last trade price of the month for historical months
                    month_end_price = float(ticker_month_trades.iloc[-1]['avg_price'])
                else:
                    # No better price available, use cost basis
                    month_end_price = position['avg_cost']
                
                # Calculate position market value
                position_value = position['shares'] * month_end_price
                position_unrealized_pnl = position_value - position['total_cost']
                
                # Add to totals
                portfolio_value += position_value
                unrealized_pnl += position_unrealized_pnl
            
            # Calculate total account value
            total_account_value = current_cash + portfolio_value
            
            # For the first month, compare to starting cash
            if not monthly_data:
                start_value = self.starting_cash
                monthly_change = total_account_value - start_value
                monthly_change_pct = (monthly_change / start_value) * 100 if start_value > 0 else 0
            else:
                # Get previous month's total account value
                previous_account_value = monthly_data[-1]['Total_Account_Value']
                
                # Calculate pure change in account value
                monthly_change = total_account_value - previous_account_value
                
                # The monthly return percentage is the change divided by previous value
                monthly_change_pct = (monthly_change / previous_account_value) * 100 if previous_account_value > 0 else 0
            
            # Record monthly summary
            month_str = str(month_period)
            monthly_data.append({
                'Month': month_str,
                'Month_Date': month_period.start_time,
                'Starting_Cash': current_cash + month_investments - month_returns,
                'Ending_Cash': current_cash,
                'Cash_Balance': current_cash,
                'Portfolio_Value': portfolio_value,
                'Realized_PnL': month_realized_pnl,
                'Unrealized_PnL': unrealized_pnl,
                'Investments': month_investments,
                'Returns': month_returns,
                'Net_Cash_Flow': month_returns - month_investments,
                'Fees': month_fees,
                'Monthly_Change': monthly_change,
                'Monthly_Change_Pct': monthly_change_pct,
                'Total_Account_Value': total_account_value
            })
        
        result_df = pd.DataFrame(monthly_data)
        return result_df
    
    def get_formatted_monthly_data(self) -> pd.DataFrame:
        """Get formatted monthly data for display"""
        monthly_df = self.calculate_monthly_balance_changes()
        if monthly_df.empty:
            return pd.DataFrame()
        
        # Create a row for starting cash
        starting_row = pd.DataFrame({
            'Month': ['Starting Cash'],
            'Total_Account_Value': [self.starting_cash],
            'Monthly_Change': [0],
            'Monthly_Change_Pct': [0]
        })
        
        # Combine starting row with monthly data
        display_df = pd.concat([starting_row, monthly_df], ignore_index=True)
        
        # Select and rename columns
        display_df = display_df[['Month', 'Total_Account_Value', 'Monthly_Change', 'Monthly_Change_Pct']]
        display_df.columns = ['Month', 'Account Value', 'Monthly Change', 'Change %']
        
        # Format numeric columns
        display_df['Account Value'] = display_df['Account Value'].apply(lambda x: format_currency(x))
        display_df['Monthly Change'] = display_df['Monthly Change'].apply(lambda x: f"${x:+,.2f}")
        display_df['Change %'] = display_df['Change %'].apply(lambda x: f"{x:+.2f}%")
        
        return display_df
    
    def calculate_trade_kpis(self) -> Dict:
        """Calculate comprehensive trading KPIs with benchmarks"""
        if self.trades_df.empty:
            return {}
        
        # Get basic stats first
        stats = self.get_summary_stats()
        
        # Use trade history for calculations
        if not hasattr(self, 'trade_history') or not self.trade_history:
            return {}
        
        # Calculate KPIs from trade history
        winning_trades = [t for t in self.trade_history if t['pnl'] > 0]
        losing_trades = [t for t in self.trade_history if t['pnl'] <= 0]
        
        total_trades = len(self.trade_history)
        if total_trades == 0:
            return {}
        
        # 1. ROI (%)
        roi_pct = (stats['total_pnl'] / stats['starting_cash']) * 100
        
        # 2. Win Rate (%)
        win_rate = (len(winning_trades) / total_trades) * 100 if total_trades > 0 else 0
        
        # 3. Average Gain
        avg_gain = sum(t['pnl'] for t in winning_trades) / len(winning_trades) if winning_trades else 0
        
        # 4. Average Loss
        avg_loss = abs(sum(t['pnl'] for t in losing_trades) / len(losing_trades)) if losing_trades else 0
        
        # 5. Risk-Reward Ratio
        rrr = avg_gain / avg_loss if avg_loss > 0 else 0
        
        # 6. Profit Factor
        gross_profit = sum(t['pnl'] for t in winning_trades) if winning_trades else 0
        gross_loss = abs(sum(t['pnl'] for t in losing_trades)) if losing_trades else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        # 7. Simplified Sharpe Ratio (using monthly returns)
        monthly_df = self.calculate_monthly_balance_changes()
        if not monthly_df.empty and len(monthly_df) > 1:
            monthly_returns = monthly_df['Monthly_Change_Pct'].dropna()
            if len(monthly_returns) > 1:
                avg_return = monthly_returns.mean()
                std_return = monthly_returns.std()
                sharpe_ratio = avg_return / std_return if std_return > 0 else 0
            else:
                sharpe_ratio = 0
        else:
            sharpe_ratio = 0
        
        # Calculate monthly ROI for benchmarking
        if not monthly_df.empty:
            months_trading = len(monthly_df)
            monthly_roi = roi_pct / months_trading if months_trading > 0 else roi_pct
        else:
            monthly_roi = roi_pct
        
        # Find best and worst trades
        if self.trade_history:
            best_trade = max(self.trade_history, key=lambda x: x['pnl'])
            worst_trade = min(self.trade_history, key=lambda x: x['pnl'])
        else:
            best_trade = None
            worst_trade = None
        
        return {
            'roi_pct': roi_pct,
            'monthly_roi': monthly_roi,
            'win_rate': win_rate,
            'avg_gain': avg_gain,
            'avg_loss': avg_loss,
            'risk_reward_ratio': rrr,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe_ratio,
            'total_trades': total_trades,
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'trade_history': self.trade_history,
            'best_trade': best_trade['pnl'] if best_trade else 0,
            'worst_trade': worst_trade['pnl'] if worst_trade else 0,
            'avg_win': avg_gain,
            'avg_loss': avg_loss,
            'largest_win': best_trade['pnl'] if best_trade else 0,
            'largest_loss': worst_trade['pnl'] if worst_trade else 0,
            'avg_hold_time': sum(t['hold_time'] for t in self.trade_history) / len(self.trade_history) if self.trade_history else 0
        }
    
    def get_kpi_rating(self, metric_name: str, value: float) -> Tuple[str, str]:
        """Get rating and color for KPI based on benchmarks"""
        if metric_name not in BENCHMARKS:
            return 'N/A', '#808080'
        
        benchmark = BENCHMARKS[metric_name]
        thresholds = benchmark['thresholds']
        ratings = benchmark['ratings']
        colors = benchmark['colors']
        
        for i, threshold in enumerate(thresholds):
            if value < threshold:
                return ratings[i], colors[i]
        
        return ratings[-1], colors[-1]
    
    def get_portfolio_allocation(self) -> Dict[str, float]:
        """Get portfolio allocation percentages"""
        portfolio_value, portfolio_details = self.calculate_portfolio_value()
        
        if portfolio_value == 0:
            return {}
        
        allocation = {}
        for ticker, details in portfolio_details.items():
            allocation[ticker] = (details['market_value'] / portfolio_value) * 100
        
        return allocation
    
    def get_sector_allocation(self) -> Dict[str, float]:
        """Get sector allocation (would need sector data integration)"""
        # This would require integration with stock info data
        # For now, return empty dict
        return {}
    
    def calculate_drawdown(self) -> Dict[str, float]:
        """Calculate drawdown statistics"""
        monthly_df = self.calculate_monthly_balance_changes()
        
        if monthly_df.empty:
            return {}
        
        # Calculate running maximum and drawdown
        account_values = monthly_df['Total_Account_Value']
        running_max = account_values.expanding().max()
        drawdown = (account_values - running_max) / running_max * 100
        
        max_drawdown = drawdown.min()
        current_drawdown = drawdown.iloc[-1]
        
        # Recovery information
        underwater_periods = (drawdown < -1).sum()  # Periods with >1% drawdown
        
        return {
            'max_drawdown': max_drawdown,
            'current_drawdown': current_drawdown,
            'underwater_periods': underwater_periods,
            'recovery_factor': abs(self.get_summary_stats()['total_return_pct'] / max_drawdown) if max_drawdown < 0 else 0
        }
    
    def export_trades_data(self) -> pd.DataFrame:
        """Export trades data for CSV download"""
        if self.trades_df.empty:
            return pd.DataFrame()
        
        export_df = self.trades_df.copy()
        export_df['trade_date'] = export_df['trade_date'].dt.strftime('%Y-%m-%d')
        
        return export_df
    
    def export_portfolio_data(self) -> pd.DataFrame:
        """Export current portfolio data for CSV download"""
        portfolio_value, portfolio_details = self.calculate_portfolio_value()
        
        if not portfolio_details:
            return pd.DataFrame()
        
        export_data = []
        for symbol, details in portfolio_details.items():
            export_data.append({
                'Symbol': symbol,
                'Shares': details['shares'],
                'Avg_Cost': details['avg_cost'],
                'Current_Price': details['current_price'],
                'Market_Value': details['market_value'],
                'Cost_Basis': details['cost_basis'],
                'Unrealized_PnL': details['unrealized_pnl'],
                'Return_Percent': details['unrealized_pnl_pct']
            })
        
        return pd.DataFrame(export_data) 