"""
AI Service for generating trading suggestions
"""
import os
import json
import time
import requests
import numpy as np
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from config.api_config import API_CONFIG
from services.strategy_manager import StrategyManager

# List of models to try in case of failure
FALLBACK_MODELS = [
    "gpt-4o-mini",
    "gpt-3.5-turbo-0125", 
    "gpt-3.5-turbo-1106",
    "gpt-3.5-turbo-16k"
]

class AIService:
    _cache = {}
    _last_request_time = 0
    _min_request_interval = 1.0  # Minimum seconds between requests
    _strategy_manager = StrategyManager()
    
    @classmethod
    def _validate_response(cls, response: Dict[str, Any]) -> bool:
        """Validate the API response structure"""
        try:
            if not isinstance(response, dict):
                return False
                
            # Check for different possible API response formats
            # Standard OpenAI format
            if 'choices' in response and response['choices']:
                # OpenAI format with message field
                if 'message' in response['choices'][0] and 'content' in response['choices'][0]['message']:
                    return True
                # Google Gemini format where content might be directly in choices
                elif 'content' in response['choices'][0]:
                    return True
                    
            # Alternate format for third-party API (free.v36.cm)
            if 'response' in response:
                return True
                
            # Another possible format
            if 'content' in response:
                return True
                
            # Check for model information for debugging
            if 'model' in response:
                print(f"Model detected: {response['model']}")
                
            return False
        except Exception as e:
            print(f"Error validating response: {str(e)}")
            return False

    @classmethod
    def clear_cache(cls, technical_data: Optional[Dict[str, Any]] = None) -> None:
        """
        Clear AI service cache.
        
        Args:
            technical_data: If provided, only clears cache for this specific data.
                           If None, clears the entire cache.
        """
        if technical_data is None:
            # Clear the entire cache
            cls._cache = {}
            print("Cache cleared completely")
        else:
            # Clear cache only for the specific technical data
            cache_key = cls._get_cache_key(technical_data)
            if cache_key in cls._cache:
                del cls._cache[cache_key]
                print(f"Cache cleared for specific technical data")
            else:
                print("No cache entry found for the provided technical data")

    @classmethod
    def _get_cache_key(cls, technical_data: Dict[str, Any]) -> str:
        """Generate a cache key from technical data"""
        # Create a simplified version of the data for caching
        if 'price' in technical_data:  # Technical analysis data
            cache_data = {
                'price': round(technical_data['price'], 2),
                'rsi': round(technical_data.get('rsi', 0), 1),
                'macd': {
                    'value': round(technical_data.get('macd', {}).get('value', 0), 2),
                    'signal': round(technical_data.get('macd', {}).get('signal', 0), 2)
                },
                'patterns': technical_data.get('patterns', [])
            }
        else:  # Performance analysis data
            cache_data = {
                'metrics': {k: round(v, 2) if isinstance(v, (int, float)) else v 
                           for k, v in technical_data.get('metrics', {}).items()},
                'time_period': technical_data.get('time_period', 'All time')
            }
        return json.dumps(cache_data, sort_keys=True)

    @classmethod
    def _check_rate_limit(cls) -> None:
        """Ensure we don't exceed rate limits"""
        current_time = time.time()
        time_since_last_request = current_time - cls._last_request_time
        
        if time_since_last_request < cls._min_request_interval:
            time.sleep(cls._min_request_interval - time_since_last_request)
        
        cls._last_request_time = time.time()

    @classmethod
    def generate_trading_suggestions(cls, technical_data: Dict[str, Any]) -> str:
        """
        Generate trading suggestions based on technical analysis data
        
        Args:
            technical_data: Dictionary containing technical indicators and patterns
            
        Returns:
            str: AI-generated trading suggestions
        """
        try:
            # Check if API is configured
            if not API_CONFIG['API_KEY'] or not API_CONFIG['API_URL']:
                return cls._generate_fallback_analysis(technical_data)
            
            # Check cache first
            cache_key = cls._get_cache_key(technical_data)
            if cache_key in cls._cache:
                cached_response = cls._cache[cache_key]
                if datetime.now() - cached_response['timestamp'] < timedelta(hours=1):
                    return cached_response['suggestions']
            
            # Always try gpt-4o-mini first, then other models if needed
            models_to_try = ["gpt-4o-mini"] + [m for m in FALLBACK_MODELS if m != "gpt-4o-mini"]
            
            for model in models_to_try:
                result = cls._try_api_request(technical_data, model)
                if result:
                    # Cache the successful response
                    cls._cache[cache_key] = {
                        'suggestions': result,
                        'timestamp': datetime.now()
                    }
                    return result
            
            # If all models failed, use the fallback analysis
            return cls._generate_fallback_analysis(technical_data)
            
        except Exception as e:
            print(f"Error generating trading suggestions: {str(e)}")
            return cls._generate_fallback_analysis(technical_data)
    
    @classmethod
    def _try_api_request(cls, technical_data: Dict[str, Any], model: str) -> Optional[str]:
        """Try making an API request with a specific model"""
        # Prepare the request
        cls._check_rate_limit()
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_CONFIG['API_KEY']}"
        }
        
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": cls._get_system_prompt()
                },
                {
                    "role": "user",
                    "content": cls._prepare_technical_analysis_prompt(technical_data)
                }
            ],
            "temperature": API_CONFIG['TEMPERATURE'],
            "max_tokens": API_CONFIG['MAX_TOKENS']
        }
        
        # Make the API request with retries
        max_retries = 2  # Reduced retries per model since we're trying multiple models
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                # Print debug info before making request
                print(f"Making API request to {API_CONFIG['API_URL']} with model {model}")
                
                response = requests.post(
                    API_CONFIG['API_URL'],
                    headers=headers,
                    json=payload,
                    timeout=API_CONFIG['REQUEST_TIMEOUT']
                )
                
                # Print response status and content for debugging
                print(f"API response status: {response.status_code}")
                try:
                    debug_content = response.text[:200] + "..." if len(response.text) > 200 else response.text
                    print(f"API response preview: {debug_content}")
                except Exception as e:
                    print(f"Could not print response preview: {str(e)}")
                
                if response.status_code == 429:  # Rate limit exceeded
                    retry_after = int(response.headers.get('Retry-After', retry_delay))
                    time.sleep(retry_after)
                    continue
                
                # For errors related to the model not being available
                if response.status_code == 503 or response.status_code == 404:
                    print(f"Model {model} not available. Trying next model.")
                    return None
                
                # Check for unauthorized messages in the response text
                if 'unauthorized' in response.text.lower() or 'auth' in response.text.lower():
                    print(f"Received 'Unauthorized request' message for model {model}. Trying next model.")
                    return None
                
                # For third-party API, handle empty or non-JSON responses
                if not response.text.strip():
                    print("Empty response received from API")
                    time.sleep(retry_delay * (attempt + 1))
                    continue
                    
                try:
                    response.raise_for_status()
                    result = response.json()
                except requests.exceptions.JSONDecodeError:
                    print(f"Invalid JSON response from API: {response.text[:100]}")
                    time.sleep(retry_delay * (attempt + 1))
                    continue
                
                # Try to extract content using different possible formats
                suggestions = None
                
                if cls._validate_response(result):
                    # Standard OpenAI format
                    if 'choices' in result and result['choices']:
                        message = result['choices'][0].get('message', {})
                        if 'content' in message:
                            suggestions = message['content']
                        # Special handling for models that return content directly in choices
                        elif 'content' in result['choices'][0]:
                            suggestions = result['choices'][0]['content']
                    # Alternate format (free.v36.cm)
                    elif 'response' in result:
                        suggestions = result['response']
                    # Another possible format
                    elif 'content' in result:
                        suggestions = result['content']
                    
                    # Check if the response contains an unauthorized message
                    if suggestions and "unauthorized request" in suggestions.lower():
                        print(f"Received 'Unauthorized request' message for model {model}. Trying next model.")
                        return None
                        
                    if suggestions:
                        print(f"Successfully generated response with model {model}")
                        return suggestions
                
                print("Valid response structure not found")
                
            except requests.exceptions.RequestException as e:
                print(f"Request exception with model {model}: {str(e)}")
                time.sleep(retry_delay * (attempt + 1))
        
        print(f"All attempts failed for model {model}")
        return None

    @classmethod
    def _get_system_prompt(cls) -> str:
        """Get the system prompt for the AI model"""
        try:
            # Get the current strategy document
            strategy_content = cls._strategy_manager.get_strategy()
            
            return f"""You are an expert stock market analyst specializing in technical analysis and trading strategies. 
Your task is to analyze technical indicators and provide clear, actionable trading suggestions based on the following strategy document:

{strategy_content}

IMPORTANT TRADING CONSTRAINTS:
- You can ONLY suggest LONG positions (buying or selling stocks)
- DO NOT suggest shorting stocks under any circumstances
- Maximum portfolio risk per trade: 5%
- Focus on two trading horizons:
  1. Swing trades (1-2 weeks)
  2. Medium-term holds (1-6 months)

When analyzing, provide:
1. Overall market sentiment
2. Trading suggestion (BUY, SELL, or HOLD)
3. Key price levels to watch
4. Risk management guidelines
5. Specific entry/exit points
6. Position sizing recommendations

Remember to:
- Be conservative in recommendations
- Always consider risk management
- Provide clear reasoning for suggestions
- Only suggest long positions (buying or selling)
- Combine multiple indicators for confirmation
- Consider both swing and medium-term opportunities"""
        except FileNotFoundError:
            # Fallback to default prompt if strategy document not found
            return """You are an expert stock market analyst specializing in technical analysis and trading strategies. 
Your task is to analyze technical indicators and provide clear, actionable trading suggestions.

IMPORTANT TRADING CONSTRAINTS:
- You can ONLY suggest LONG positions (buying or selling stocks)
- DO NOT suggest shorting stocks under any circumstances
- Maximum portfolio risk per trade: 5%

When analyzing, provide:
1. Overall market sentiment
2. Trading suggestion (BUY, SELL, or HOLD)
3. Key price levels to watch
4. Risk management guidelines

Remember to:
- Be conservative in recommendations
- Always consider risk management
- Provide clear reasoning for suggestions
- Only suggest long positions (buying or selling)"""

    @classmethod
    def _prepare_technical_analysis_prompt(cls, technical_data: Dict[str, Any]) -> str:
        """Prepare the prompt for technical analysis"""
        # Get the basic information from technical data
        price = technical_data.get('price', 0)
        macd_value = technical_data.get('macd', {}).get('value', 0)
        macd_signal = technical_data.get('macd', {}).get('signal', 0)
        rsi = technical_data.get('rsi', 0)
        stoch_k = technical_data.get('stochastic', {}).get('k', 0)
        stoch_d = technical_data.get('stochastic', {}).get('d', 0)
        adx = technical_data.get('adx', 0)
        volume_ratio = technical_data.get('volume_ratio', 0)
        bb_position = technical_data.get('bb_position', 0.5)
        atr = technical_data.get('atr', 0)
        patterns = technical_data.get('patterns', [])
            
        # Get SMA values if available
        sma_20 = technical_data.get('sma_20', 0)
        sma_50 = technical_data.get('sma_50', 0)
        sma_200 = technical_data.get('sma_200', 0)
        
        # Format all indicators into a comprehensive technical analysis
        prompt = f"""Please analyze this stock based on the following technical indicators:

PRICE & MOVING AVERAGES:
- Current Price: ${price:.2f}
- SMA 20: ${sma_20:.2f}
- SMA 50: ${sma_50:.2f}
- SMA 200: ${sma_200:.2f}
- Price vs SMA 20: {'Above' if price > sma_20 else 'Below'}
- Price vs SMA 50: {'Above' if price > sma_50 else 'Below'}
- Price vs SMA 200: {'Above' if price > sma_200 else 'Below'}
- Golden Cross (SMA 50 > SMA 200): {'Yes' if sma_50 > sma_200 else 'No'}

MOMENTUM INDICATORS:
- RSI (14): {rsi:.1f}
- MACD: {macd_value:.3f}
- MACD Signal: {macd_signal:.3f}
- MACD Histogram: {(macd_value - macd_signal):.3f}
- Stochastic %K: {stoch_k:.1f}
- Stochastic %D: {stoch_d:.1f}
- ADX: {adx:.1f}
- Bollinger Band Position: {bb_position:.2f} (0=lower band, 1=upper band, 0.5=middle)

VOLATILITY & VOLUME:
- ATR: {atr:.2f}
- Volume Ratio (Current/SMA): {volume_ratio:.2f}

DETECTED PATTERNS:
{', '.join(patterns) if patterns else 'None detected'}

Based on these indicators, please provide:
1. Comprehensive technical analysis (bullish/bearish factors)
2. Overall sentiment (strongly bullish, mildly bullish, neutral, mildly bearish, strongly bearish)
3. Trading suggestion (BUY, SELL, or HOLD)
4. Key price levels to watch (support, resistance)
5. Entry/exit strategy for both swing and medium-term perspectives
6. Position sizing recommendation
7. Stop-loss placement

Please provide concrete analysis, not general statements about indicators."""
        
        return prompt

    @classmethod
    def _generate_fallback_analysis(cls, technical_data: Dict[str, Any]) -> str:
        """Generate a fallback analysis when API is unavailable"""
        # Get basic technical indicators
        price = technical_data.get('price', 0)
        rsi = technical_data.get('rsi', 50)
        macd_value = technical_data.get('macd', {}).get('value', 0)
        macd_signal = technical_data.get('macd', {}).get('signal', 0)
        stoch_k = technical_data.get('stochastic', {}).get('k', 50)
        stoch_d = technical_data.get('stochastic', {}).get('d', 50)
        adx = technical_data.get('adx', 20)
        
        # Simple sentiment analysis
        bullish_factors = []
        bearish_factors = []
        
        # RSI analysis
        if rsi < 30:
            bullish_factors.append("RSI shows oversold conditions")
        elif rsi > 70:
            bearish_factors.append("RSI shows overbought conditions")
            
        # MACD analysis
        if macd_value > macd_signal:
            bullish_factors.append("MACD is above signal line")
        else:
            bearish_factors.append("MACD is below signal line")
            
        # Stochastic analysis
        if stoch_k < 20:
            bullish_factors.append("Stochastic shows oversold conditions")
        elif stoch_k > 80:
            bearish_factors.append("Stochastic shows overbought conditions")
            
        if stoch_k > stoch_d:
            bullish_factors.append("Stochastic %K crossed above %D")
        else:
            bearish_factors.append("Stochastic %K crossed below %D")
            
        # ADX analysis
        if adx > 25:
            if len(bullish_factors) > len(bearish_factors):
                bullish_factors.append("Strong trend strength")
            elif len(bearish_factors) > len(bullish_factors):
                bearish_factors.append("Strong trend strength")
        
        # Determine overall sentiment
        if len(bullish_factors) > len(bearish_factors) + 1:
            sentiment = "mildly bullish" if len(bearish_factors) else "strongly bullish"
            recommendation = "BUY"
        elif len(bearish_factors) > len(bullish_factors) + 1:
            sentiment = "mildly bearish" if len(bullish_factors) else "strongly bearish"
            recommendation = "SELL" if len(bearish_factors) > len(bullish_factors) + 2 else "HOLD"
        else:
            sentiment = "neutral"
            recommendation = "HOLD"
        
        # Generate the analysis
        analysis = f"""# Technical Analysis

## Overall Assessment
The technical indicators suggest a {sentiment} outlook.

## Bullish Factors
{chr(10).join("- " + factor for factor in bullish_factors) if bullish_factors else "None identified"}

## Bearish Factors
{chr(10).join("- " + factor for factor in bearish_factors) if bearish_factors else "None identified"}

## Trading Suggestion
**{recommendation}**

## Key Price Levels
- Current Price: ${price:.2f}
- Support: ${price * 0.95:.2f} (estimated)
- Resistance: ${price * 1.05:.2f} (estimated)

## Risk Management
- Position Size: 2-3% of portfolio
- Stop Loss: ${price * 0.93:.2f} (7% below current price)
- Take Profit: ${price * 1.12:.2f} (12% above current price)

*Note: This is an automated analysis. Please use it in conjunction with your own research.*"""
        
        return analysis

    @classmethod
    def generate_performance_analysis(cls, technical_data: Dict[str, Any]) -> str:
        """
        Generate performance analysis based on portfolio metrics
        
        Args:
            technical_data: Dictionary containing performance metrics and time period
            
        Returns:
            str: AI-generated performance analysis
        """
        try:
            # Check if API is configured
            if not API_CONFIG['API_KEY'] or not API_CONFIG['API_URL']:
                return cls._generate_fallback_performance_analysis(technical_data)
            
            # Check cache first
            cache_key = cls._get_cache_key(technical_data)
            if cache_key in cls._cache:
                cached_response = cls._cache[cache_key]
                if datetime.now() - cached_response['timestamp'] < timedelta(hours=1):
                    return cached_response['suggestions']
            
            # Always try gpt-4o-mini first, then other models if needed
            models_to_try = ["gpt-4o-mini"] + [m for m in FALLBACK_MODELS if m != "gpt-4o-mini"]
            
            for model in models_to_try:
                result = cls._try_performance_api_request(technical_data, model)
                if result:
                    # Cache the successful response
                    cls._cache[cache_key] = {
                        'suggestions': result,
                        'timestamp': datetime.now()
                    }
                    return result
            
            # If all models failed, use the fallback analysis
            return cls._generate_fallback_performance_analysis(technical_data)
            
        except Exception as e:
            print(f"Error generating performance analysis: {str(e)}")
            return cls._generate_fallback_performance_analysis(technical_data)
    
    @classmethod
    def _try_performance_api_request(cls, technical_data: Dict[str, Any], model: str) -> Optional[str]:
        """Try making an API request for performance analysis with a specific model"""
        # Prepare the request
        cls._check_rate_limit()
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_CONFIG['API_KEY']}"
        }
        
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": cls._get_performance_system_prompt()
                },
                {
                    "role": "user",
                    "content": cls._prepare_performance_prompt(technical_data)
                }
            ],
            "temperature": API_CONFIG['TEMPERATURE'],
            "max_tokens": API_CONFIG['MAX_TOKENS']
        }
        
        # Make the API request with retries
        max_retries = 2  # Reduced retries per model since we're trying multiple models
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                # Print debug info before making request
                print(f"Making performance API request to {API_CONFIG['API_URL']} with model {model}")
                
                response = requests.post(
                    API_CONFIG['API_URL'],
                    headers=headers,
                    json=payload,
                    timeout=API_CONFIG['REQUEST_TIMEOUT']
                )
                
                # Print response status and content for debugging
                print(f"API response status: {response.status_code}")
                try:
                    debug_content = response.text[:200] + "..." if len(response.text) > 200 else response.text
                    print(f"API response preview: {debug_content}")
                except Exception as e:
                    print(f"Could not print response preview: {str(e)}")
                
                if response.status_code == 429:  # Rate limit exceeded
                    retry_after = int(response.headers.get('Retry-After', retry_delay))
                    time.sleep(retry_after)
                    continue
                
                # For errors related to the model not being available
                if response.status_code == 503 or response.status_code == 404:
                    print(f"Model {model} not available for performance analysis. Trying next model.")
                    return None
                
                # For third-party API, handle empty or non-JSON responses
                if not response.text.strip():
                    print("Empty response received from API")
                    time.sleep(retry_delay * (attempt + 1))
                    continue
                    
                try:
                    response.raise_for_status()
                    result = response.json()
                except requests.exceptions.JSONDecodeError:
                    print(f"Invalid JSON response from API: {response.text[:100]}")
                    time.sleep(retry_delay * (attempt + 1))
                    continue
                
                # Try to extract content using different possible formats
                analysis = None
                
                if cls._validate_response(result):
                    # Standard OpenAI format
                    if 'choices' in result and result['choices']:
                        message = result['choices'][0].get('message', {})
                        if 'content' in message:
                            analysis = message['content']
                        # Special handling for models that return content directly in choices
                        elif 'content' in result['choices'][0]:
                            analysis = result['choices'][0]['content']
                    # Alternate format (free.v36.cm)
                    elif 'response' in result:
                        analysis = result['response']
                    # Another possible format
                    elif 'content' in result:
                        analysis = result['content']
                    
                    # Check if the response contains an unauthorized message
                    if analysis and "unauthorized request" in analysis.lower():
                        print(f"Received 'Unauthorized request' message for model {model}. Trying next model.")
                        return None
                        
                    if analysis:
                        print(f"Successfully generated performance analysis with model {model}")
                        return analysis
                
                print("Valid response structure not found")
                
            except requests.exceptions.RequestException as e:
                print(f"Request exception with model {model} for performance analysis: {str(e)}")
                time.sleep(retry_delay * (attempt + 1))
        
        print(f"All attempts failed for model {model}")
        return None

    @classmethod
    def _get_performance_system_prompt(cls) -> str:
        """Get the system prompt for performance analysis"""
        return """You are an expert trading performance analyst and portfolio manager. Your task is to provide a comprehensive analysis of trading performance data and deliver actionable insights for improvement.

When analyzing trading performance, consider these key areas:

1. Trading Insights:
   - Identify significant patterns in trading success and failure
   - Assess strategy adherence and effectiveness
   - Evaluate risk-adjusted returns
   - Analyze hold time effectiveness
   - Detect potential performance issues or hidden risks
   - Recognize improvements or deteriorations in performance over time
   - Consider market conditions impact on performance

2. Recommendations:
   - Provide specific, actionable steps to improve trading performance
   - Suggest position sizing adjustments based on success patterns
   - Recommend risk management enhancements
   - Identify optimal entry and exit timing strategies
   - Suggest portfolio allocation improvements
   - Recommend specific trading behaviors to continue or avoid
   - Highlight psychological aspects of trading that need attention

3. Risk Assessment:
   - Evaluate current portfolio risk levels
   - Assess concentration risks
   - Calculate potential drawdown scenarios
   - Review risk-reward profile sustainability
   - Consider market condition impacts on current positions
   - Analyze risk-adjusted returns across different time frames
   - Identify potential systemic risks in the trading approach

IMPORTANT: Format your response EXACTLY as follows:

Trading Insights:
- [Your first insight]
- [Your second insight]
- [Additional insights...]

Recommendations:
- [Your first recommendation]
- [Your second recommendation]
- [Additional recommendations...]

Risk Assessment:
- [Your first risk assessment point]
- [Your second risk assessment point]
- [Additional risk points...]

Each item must:
1. Start with a bullet point (using - )
2. Be specific, detailed and actionable where appropriate
3. Focus on the trader's actual performance data
4. Be based on established trading principles
5. Prioritize practical advice over general statements
6. Include both strengths to maintain and weaknesses to improve"""

    @classmethod
    def _prepare_performance_prompt(cls, technical_data: Dict[str, Any]) -> str:
        """Prepare the prompt for performance analysis"""
        metrics = technical_data.get('metrics', {})
        strategy_content = technical_data.get('strategy', '')
        portfolio_stats = technical_data.get('portfolio_stats', {})
        patterns = technical_data.get('patterns', {})
        time_period = technical_data.get('time_period', 'All time')
        
        # Create a comprehensive prompt with all available data
        prompt = f"""Generate a comprehensive trading performance analysis for {time_period}.

Performance Metrics:
- Win Rate: {metrics.get('win_rate', 0):.1f}%
- Risk-Reward Ratio: {metrics.get('risk_reward_ratio', 0):.2f}
- Profit Factor: {metrics.get('profit_factor', 0):.2f}
- Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}
- Average Hold Time: {metrics.get('avg_hold_time', 0):.1f} days
- Total Trades: {metrics.get('total_trades', 0)}
- Average Win: ${metrics.get('avg_win', 0):.2f}
- Average Loss: ${metrics.get('avg_loss', 0):.2f}
- Total ROI: {metrics.get('roi_pct', 0):.2f}%
- Monthly ROI: {metrics.get('monthly_roi', 0):.2f}%

Portfolio Statistics:
- Total Account Value: ${portfolio_stats.get('total_account_value', 0):,.2f}
- Portfolio Value: ${portfolio_stats.get('portfolio_value', 0):,.2f}
- Cash Balance: ${portfolio_stats.get('cash_balance', 0):,.2f}
- Realized P&L: ${portfolio_stats.get('realized_pnl', 0):,.2f}
- Unrealized P&L: ${portfolio_stats.get('unrealized_pnl', 0):,.2f}
- Total Return: {portfolio_stats.get('total_return_pct', 0):.2f}%"""

        # Add trading pattern information if available
        if patterns:
            prompt += f"""

Trading Patterns:
- Symbols Traded: {patterns.get('symbols_traded', 0)}
- Hold Time Distribution:
  * Short-term trades (≤5 days): {patterns.get('hold_time_distribution', {}).get('short_term', 0)}
  * Medium-term trades (6-20 days): {patterns.get('hold_time_distribution', {}).get('medium_term', 0)}
  * Long-term trades (>20 days): {patterns.get('hold_time_distribution', {}).get('long_term', 0)}
- Win Rate by Hold Time:
  * Short-term: {patterns.get('win_rate_by_hold_time', {}).get('short_term', 0):.1f}%
  * Medium-term: {patterns.get('win_rate_by_hold_time', {}).get('medium_term', 0):.1f}%
  * Long-term: {patterns.get('win_rate_by_hold_time', {}).get('long_term', 0):.1f}%"""

        # Add trading strategy if available
        if strategy_content:
            # Limit strategy length to avoid token overflow
            max_strategy_length = 1000
            if len(strategy_content) > max_strategy_length:
                strategy_content = strategy_content[:max_strategy_length] + "... (truncated)"
                
            prompt += f"""

Trading Strategy:
{strategy_content}"""

        # Add analysis requirements
        prompt += """

Based on this information, please provide:

1. Trading Insights:
   - Deep analysis of strengths and weaknesses
   - Pattern recognition in winning vs losing trades
   - Hold time optimization opportunities
   - Strategy effectiveness assessment

2. Recommendations:
   - Specific, actionable improvements for trading approach
   - Risk management enhancements
   - Position sizing suggestions
   - Entry/exit timing optimization

3. Risk Assessment:
   - Portfolio risk evaluation
   - Concentration risk
   - Drawdown vulnerability
   - Risk-adjusted return potential

Focus on providing specific, actionable insights that can help improve trading performance."""

        return prompt

    @classmethod
    def _generate_fallback_performance_analysis(cls, technical_data: Dict[str, Any]) -> str:
        """Generate a fallback performance analysis when API is unavailable"""
        metrics = technical_data.get('metrics', {})
        
        # Calculate risk level based on metrics
        win_rate = metrics.get('win_rate', 0)
        risk_reward = metrics.get('risk_reward_ratio', 0)
        profit_factor = metrics.get('profit_factor', 0)
        sharpe = metrics.get('sharpe_ratio', 0)
        
        # Determine risk level
        if win_rate >= 60 and risk_reward >= 2 and profit_factor >= 1.5 and sharpe >= 1.5:
            risk_level = "Low Risk"
            risk_description = "Your trading shows strong risk management and consistent returns."
        elif win_rate >= 50 and risk_reward >= 1.5 and profit_factor >= 1.2 and sharpe >= 1.0:
            risk_level = "Moderate Risk"
            risk_description = "Your risk-adjusted returns are acceptable but could be improved."
        else:
            risk_level = "High Risk"
            risk_description = "Your trading shows elevated risk levels that need attention."
        
        # Generate dynamic recommendations based on metrics
        recommendations = []
        if win_rate < 50:
            recommendations.append("Focus on improving your win rate through better entry timing and trade selection")
        if risk_reward < 1.5:
            recommendations.append("Work on maintaining a higher risk-reward ratio in your trades")
        if profit_factor < 1.2:
            recommendations.append("Review your trade management to improve profit factor")
        if sharpe < 1.0:
            recommendations.append("Consider adjusting position sizing to improve risk-adjusted returns")
        
        # If no specific recommendations, provide general ones
        if not recommendations:
            recommendations = [
                "Continue monitoring your trading metrics",
                "Maintain your current risk management approach",
                "Consider documenting your successful trade patterns"
            ]
        
        return f"""Trading Insights:
- Win Rate: {win_rate:.1f}%
- Risk-Reward Ratio: {risk_reward:.2f}
- Profit Factor: {profit_factor:.2f}
- Sharpe Ratio: {sharpe:.2f}

Recommendations:
{chr(10).join('- ' + rec for rec in recommendations)}

Risk Assessment:
- {risk_level}: {risk_description}""" 