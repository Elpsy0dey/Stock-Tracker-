# Deposit Tracking Feature

This PR adds deposit tracking functionality to the Stock Tracker application, allowing users to:

1. Add deposits to their portfolio with specific dates
2. View a list of all deposits in both the sidebar and portfolio overview
3. See total deposits as part of the portfolio metrics
4. Calculate returns more accurately by accounting for deposits

## Changes

### Models/portfolio_tracker.py
- Added deposit tracking properties (`deposits` list and `total_deposits` sum)
- Added `add_deposit()` method to record new deposits
- Updated cash balance calculation to include deposits
- Updated return calculation to account for deposits
- Added deposits to summary statistics

### Main.py
- Added deposit management UI in the sidebar
- Added deposit display section in the portfolio overview
- Added deposit metrics to the portfolio summary

## How to Use

1. Open the "Manage Deposits" expander in the sidebar
2. Enter a deposit amount and date
3. Click "Add Deposit" to record the deposit
4. View all deposits and totals in both the sidebar and portfolio overview

## Benefits

- More accurate tracking of portfolio performance
- Better distinction between returns from trading vs. additional capital
- Improved cash flow tracking
- More realistic performance metrics

## Example

If a user starts with $50,000 and later adds a $10,000 deposit, the application will:
- Track the $10,000 deposit separately
- Show total deposits of $10,000
- Calculate returns based on the combined $60,000 investment
- Show the correct cash balance including the deposit 