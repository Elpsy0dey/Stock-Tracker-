"""
Test script for AI service
"""
from services.ai_service import AIService
from datetime import datetime, timedelta

def test_technical_analysis():
    """Test the technical analysis AI function"""
    print("\n\n=== Testing Technical Analysis AI ===\n")

    # Test data for technical analysis
    technical_data = {
        'price': 100.0,
        'rsi': 65.0,
        'macd': {
            'value': 0.5,
            'signal': 0.3
        },
        'stochastic': {
            'k': 75.0,
            'd': 70.0
        },
        'adx': 30.0,
        'volume_ratio': 1.5,
        'bb_position': 0.7,
        'atr': 2.0,
        'patterns': ['Ascending Triangle'],
        'sma_20': 98.0,
        'sma_50': 95.0,
        'sma_200': 90.0
    }

    try:
        # Generate suggestions
        suggestions = AIService.generate_trading_suggestions(technical_data)
        print("\nAI Trading Suggestions:")
        print("=" * 80)
        print(suggestions)
        print("=" * 80)
        print("\nTechnical analysis test completed successfully!")
        
    except Exception as e:
        print(f"\nError testing technical analysis: {str(e)}")

def test_performance_analysis():
    """Test the performance analysis AI function"""
    print("\n\n=== Testing Performance Analysis AI ===\n")
    
    # Create sample trade history
    sample_trades = [
        {
            'symbol': 'AAPL',
            'entry_date': datetime.now() - timedelta(days=30),
            'exit_date': datetime.now() - timedelta(days=25),
            'entry_price': 150.0,
            'exit_price': 165.0,
            'units': 10,
            'pnl': 150.0,
            'pnl_pct': 10.0,
            'hold_time': 5,
            'fees': 10.0
        },
        {
            'symbol': 'MSFT',
            'entry_date': datetime.now() - timedelta(days=20),
            'exit_date': datetime.now() - timedelta(days=10),
            'entry_price': 320.0,
            'exit_price': 310.0,
            'units': 5,
            'pnl': -50.0,
            'pnl_pct': -3.13,
            'hold_time': 10,
            'fees': 10.0
        },
        {
            'symbol': 'GOOGL',
            'entry_date': datetime.now() - timedelta(days=15),
            'exit_date': datetime.now() - timedelta(days=5),
            'entry_price': 140.0,
            'exit_price': 148.0,
            'units': 15,
            'pnl': 120.0,
            'pnl_pct': 5.71,
            'hold_time': 10,
            'fees': 10.0
        }
    ]
    
    # Convert datetime objects to strings for serialization
    for trade in sample_trades:
        trade['entry_date'] = trade['entry_date'].strftime('%Y-%m-%d')
        trade['exit_date'] = trade['exit_date'].strftime('%Y-%m-%d')
    
    # Sample performance data
    performance_data = {
        'trade_history': sample_trades,
        'metrics': {
            'win_rate': 66.67,
            'risk_reward_ratio': 1.8,
            'profit_factor': 2.7,
            'sharpe_ratio': 1.4,
            'avg_hold_time': 8.3,
            'total_trades': 3,
            'avg_win': 135.0,
            'avg_loss': 50.0,
            'roi_pct': 7.33,
            'monthly_roi': 2.44
        },
        'time_period': 'Last month',
        'portfolio_stats': {
            'total_account_value': 12500.0,
            'portfolio_value': 8500.0,
            'cash_balance': 4000.0,
            'realized_pnl': 220.0,
            'unrealized_pnl': 150.0,
            'total_return_pct': 8.2
        },
        'patterns': {
            'symbols_traded': 3,
            'hold_time_distribution': {
                'short_term': 1,
                'medium_term': 2,
                'long_term': 0
            },
            'win_rate_by_hold_time': {
                'short_term': 100.0,
                'medium_term': 50.0,
                'long_term': 0.0
            }
        }
    }
    
    # Add a sample strategy snippet
    sample_strategy = """
    Trading Strategy Summary:
    
    1. Entry Criteria:
       - RSI below 30 for oversold conditions
       - Price above 50-day moving average
       - MACD histogram showing positive divergence
    
    2. Exit Criteria:
       - RSI above 70 for overbought conditions
       - Price target at 15% gain
       - Stop loss at 7% below entry price
    
    3. Risk Management:
       - Maximum 5% risk per trade
       - Position sizing based on volatility
       - Diversification across sectors
    """
    
    performance_data['strategy'] = sample_strategy
    
    try:
        # Generate performance analysis
        analysis = AIService.generate_performance_analysis(performance_data)
        print("\nAI Performance Analysis:")
        print("=" * 80)
        print(analysis)
        print("=" * 80)
        print("\nPerformance analysis test completed successfully!")
        
    except Exception as e:
        print(f"\nError testing performance analysis: {str(e)}")

if __name__ == "__main__":
    test_technical_analysis()
    test_performance_analysis() 