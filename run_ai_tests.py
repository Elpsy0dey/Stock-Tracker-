#!/usr/bin/env python
"""
Script to run AI tests for the Stock Tracker application

This script tests both AI components:
1. Technical Analysis AI
2. Performance Analysis AI

Usage:
- Run the entire test suite:
  python run_ai_tests.py

- Run only technical analysis test:
  python run_ai_tests.py technical

- Run only performance analysis test:
  python run_ai_tests.py performance
"""

import sys
from test_ai_service import test_technical_analysis, test_performance_analysis

def print_header():
    print("\n" + "=" * 80)
    print(" STOCK TRACKER AI TESTS ")
    print("=" * 80)
    print("Testing AI functionality to ensure prompts generate properly")
    print("=" * 80)

def print_footer():
    print("\n" + "=" * 80)
    print(" AI TESTS COMPLETED ")
    print("=" * 80)

def main():
    print_header()
    
    # Determine which tests to run based on command-line arguments
    if len(sys.argv) > 1:
        test_type = sys.argv[1].lower()
        
        if test_type == "technical":
            print("\nRunning only Technical Analysis AI test...")
            test_technical_analysis()
        elif test_type == "performance":
            print("\nRunning only Performance Analysis AI test...")
            test_performance_analysis()
        else:
            print(f"\nUnknown test type: {test_type}")
            print("Valid options: technical, performance")
            print("Running all tests instead...")
            test_technical_analysis()
            test_performance_analysis()
    else:
        # Run all tests
        test_technical_analysis()
        test_performance_analysis()
    
    print_footer()

if __name__ == "__main__":
    main() 