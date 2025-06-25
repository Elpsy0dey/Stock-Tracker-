"""
Unit tests for the deposit tracking feature
"""

import unittest
import pandas as pd
from datetime import datetime, timedelta
from models.portfolio_tracker import PortfolioTracker

class TestDepositTracking(unittest.TestCase):
    """Test cases for deposit tracking functionality"""
    
    def setUp(self):
        """Set up test environment before each test"""
        self.portfolio_tracker = PortfolioTracker(starting_cash=10000.0)
        
        # Create sample trades data
        self.trades_data = pd.DataFrame({
            'trade_date': [datetime.now() - timedelta(days=10), datetime.now() - timedelta(days=5)],
            'ticker': ['AAPL', 'AAPL'],
            'side': ['BUY', 'SELL'],
            'units': [10, 5],
            'avg_price': [150.0, 160.0],
            'fees': [5.0, 5.0]
        })
        
        self.portfolio_tracker.load_trades(self.trades_data)
    
    def test_add_deposit(self):
        """Test adding deposits"""
        # Add a deposit
        result = self.portfolio_tracker.add_deposit(1000.0)
        self.assertTrue(result)
        self.assertEqual(len(self.portfolio_tracker.deposits), 1)
        self.assertEqual(self.portfolio_tracker.total_deposits, 1000.0)
        
        # Add another deposit
        result = self.portfolio_tracker.add_deposit(2000.0)
        self.assertTrue(result)
        self.assertEqual(len(self.portfolio_tracker.deposits), 2)
        self.assertEqual(self.portfolio_tracker.total_deposits, 3000.0)
        
        # Try adding an invalid deposit
        result = self.portfolio_tracker.add_deposit(-100.0)
        self.assertFalse(result)
        self.assertEqual(len(self.portfolio_tracker.deposits), 2)
        self.assertEqual(self.portfolio_tracker.total_deposits, 3000.0)
    
    def test_cash_balance_with_deposits(self):
        """Test cash balance calculation with deposits"""
        # Initial cash balance without deposits
        initial_cash = self.portfolio_tracker.calculate_cash_balance()
        
        # Add a deposit
        deposit_amount = 5000.0
        self.portfolio_tracker.add_deposit(deposit_amount)
        
        # Cash balance should increase by deposit amount
        new_cash = self.portfolio_tracker.calculate_cash_balance()
        self.assertEqual(new_cash, initial_cash + deposit_amount)
    
    def test_summary_stats_with_deposits(self):
        """Test summary statistics with deposits"""
        # Add a deposit
        deposit_amount = 2500.0
        self.portfolio_tracker.add_deposit(deposit_amount)
        
        # Get summary stats
        stats = self.portfolio_tracker.get_summary_stats()
        
        # Check deposit-related stats
        self.assertEqual(stats['total_deposits'], deposit_amount)
        self.assertTrue('deposits' in stats)
        self.assertEqual(len(stats['deposits']), 1)
        
        # Check that return calculation accounts for deposits
        expected_return_pct = ((stats['total_account_value'] - stats['starting_cash'] - stats['total_deposits']) / 
                              (stats['starting_cash'] + stats['total_deposits'])) * 100
        self.assertAlmostEqual(stats['total_return_pct'], expected_return_pct, places=5)

if __name__ == '__main__':
    unittest.main() 