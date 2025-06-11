# Stock Tracker

A comprehensive web-based application for portfolio tracking, technical analysis, stock screening, risk management, and AI-powered trading signals.

## 🚀 Features

- **📊 Portfolio Tracking**: Track your trades, positions, and overall performance
- **📈 Technical Analysis**: 88+ indicators for comprehensive market analysis
- **🎯 Intelligent Stock Screening**: Find swing and breakout opportunities
- **⚖️ Advanced Risk Management**: Position sizing and risk assessment tools
- **🤖 AI-Powered Analysis**: Trading suggestions and market insights powered by advanced AI
- **📉 Backtesting**: Test strategies against historical market data

## 🔍 Key Components

### Portfolio Tracker
- Position management
- Performance metrics
- Trade history analysis

### Technical Analysis
- Multiple timeframe analysis
- Pattern recognition
- Support/resistance identification

### Stock Screener
- Swing trading opportunities (1-2 week horizon)
- Breakout trading setups (1-6 month horizon)
- Research-backed screening criteria

### AI Integration
- **NEW**: Automatic AI model selection
- Trading suggestions based on technical indicators
- Performance analysis and improvement recommendations

## ⚙️ Recent Enhancements

### AI Model Auto-Selection Feature
- Automatically checks which AI models are available at startup
- Selects the best available model for optimal performance
- Allows manual model selection through settings interface
- Fallback mechanisms ensure continuous operation

## 🛠️ Technology Stack

- **Backend**: Python with Streamlit
- **Data Analysis**: Pandas, NumPy
- **Visualization**: Plotly
- **AI Integration**: OpenAI API with custom model selection
- **Market Data**: yfinance API

## 📋 Getting Started

### Prerequisites
- Python 3.8+
- Git

### Installation

1. Clone the repository:
```
git clone https://github.com/Elpsy0dey/Stock-Tracker-.git
cd Stock-Tracker-
```

2. Install dependencies:
```
pip install -r requirements.txt
```

3. Configure API settings:
   - Create a `.env` file with your API credentials (for local development)
   - Or use the Streamlit secrets approach (see below)

4. Run the application:
```
python main.py
```

## 🔐 API Credentials Management

### Local Development

For local development, you can use either:

1. **Environment Variables (.env file)**
   - Create a `.env` file in the project root
   - Add your API credentials:
     ```
     OPENAI_API_KEY=your_api_key_here
     OPENAI_API_URL=https://free.v36.cm/v1/chat/completions
     ```

2. **Streamlit Secrets**
   - Create a `.streamlit/secrets.toml` file
   - Use the provided template from `.streamlit/secrets.toml.example`
   - Add your actual credentials

### Deploying to Streamlit Cloud

When deploying to Streamlit Cloud, your credentials should be stored in Streamlit Secrets:

1. Deploy your app to Streamlit Cloud
2. Go to your app's settings
3. In the "Secrets" section, add your credentials in TOML format:
   ```toml
   OPENAI_API_KEY = "your_api_key_here"
   OPENAI_API_URL = "https://free.v36.cm/v1/chat/completions"
   OPENAI_MODEL = "gpt-4o-mini"
   ```
4. Save changes

This approach keeps your credentials secure and out of version control.

## 📊 Use Cases

- Individual investors tracking portfolio performance
- Traders looking for technical setups
- Strategy backtesting and optimization
- Risk management and position sizing

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

Based on research studies for high win-rate trading strategies and technical analysis.