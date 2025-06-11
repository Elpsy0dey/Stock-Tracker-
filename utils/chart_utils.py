"""
Chart utilities for the Trading Portfolio Tracker

Creates interactive charts and visualizations using Plotly
"""

import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Optional
import streamlit as st
from config.settings import CHART_HEIGHT, CHART_TEMPLATE

def create_monthly_balance_chart(monthly_df: pd.DataFrame) -> go.Figure:
    """Create monthly balance change chart"""
    if monthly_df.empty:
        return None
    
    # Create figure
    fig = go.Figure()
    
    # Add total account value line
    fig.add_trace(go.Scatter(
        x=monthly_df['Month'],
        y=monthly_df['Total_Account_Value'],
        mode='lines+markers',
        name='Total Account Value',
        line=dict(color='#2E86AB', width=3),
        marker=dict(size=8),
        hovertemplate='<b>%{x}</b><br>Account Value: $%{y:,.2f}<extra></extra>'
    ))
    
    fig.update_layout(
        title='Monthly Account Value Tracking',
        xaxis_title='Month',
        yaxis_title='Account Value ($)',
        hovermode='x unified',
        template=CHART_TEMPLATE,
        height=CHART_HEIGHT,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    return fig

def create_portfolio_allocation_chart(portfolio_details: Dict) -> go.Figure:
    """Create portfolio allocation pie chart"""
    if not portfolio_details:
        return None
    
    symbols = list(portfolio_details.keys())
    values = [details['market_value'] for details in portfolio_details.values()]
    
    fig = px.pie(
        values=values, 
        names=symbols, 
        title="Portfolio Allocation by Market Value"
    )
    
    fig.update_layout(
        template=CHART_TEMPLATE,
        height=CHART_HEIGHT
    )
    
    return fig

def create_technical_analysis_chart(df: pd.DataFrame, symbol: str) -> go.Figure:
    """Create comprehensive technical analysis chart"""
    if df.empty:
        return None
    
    # Create subplots
    fig = make_subplots(
        rows=5, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.4, 0.15, 0.15, 0.15, 0.15],
        subplot_titles=(
            f'{symbol} - Price & Technical Indicators',
            'Volume & OBV',
            'Momentum Indicators',
            'Trend Indicators',
            'Volatility Bands'
        )
    )
    
    # Main price chart with Bollinger Bands
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='Price'
        ),
        row=1, col=1
    )
    
    # Bollinger Bands
    if 'BB_Upper' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['BB_Upper'],
                name='BB Upper',
                line=dict(color='rgba(250,250,250,0.5)', width=1),
                showlegend=False
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['BB_Lower'],
                name='BB Lower',
                line=dict(color='rgba(250,250,250,0.5)', width=1),
                fill='tonexty',
                fillcolor='rgba(250,250,250,0.2)',
                showlegend=False
            ),
            row=1, col=1
        )
    
    # Moving Averages
    if 'SMA_20' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['SMA_20'],
                name='SMA 20',
                line=dict(color='blue', width=1)
            ),
            row=1, col=1
        )
    
    if 'SMA_50' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['SMA_50'],
                name='SMA 50',
                line=dict(color='orange', width=1)
            ),
            row=1, col=1
        )
    
    if 'SMA_200' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['SMA_200'],
                name='SMA 200',
                line=dict(color='red', width=1)
            ),
            row=1, col=1
        )
    
    # Volume and OBV
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df['Volume'],
            name='Volume',
            marker_color='rgba(0,0,255,0.5)'
        ),
        row=2, col=1
    )
    
    if 'OBV' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['OBV'],
                name='OBV',
                line=dict(color='green', width=1)
            ),
            row=2, col=1
        )
    
    # Momentum Indicators
    if 'RSI' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['RSI'],
                name='RSI',
                line=dict(color='purple', width=1)
            ),
            row=3, col=1
        )
        
        # Add RSI levels
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
    
    if 'Stoch_K' in df.columns and 'Stoch_D' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['Stoch_K'],
                name='Stoch %K',
                line=dict(color='blue', width=1)
            ),
            row=3, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['Stoch_D'],
                name='Stoch %D',
                line=dict(color='red', width=1)
            ),
            row=3, col=1
        )
    
    # Trend Indicators
    if 'MACD' in df.columns and 'MACD_Signal' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['MACD'],
                name='MACD',
                line=dict(color='blue', width=1)
            ),
            row=4, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['MACD_Signal'],
                name='MACD Signal',
                line=dict(color='red', width=1)
            ),
            row=4, col=1
        )
        
        if 'MACD_Histogram' in df.columns:
            fig.add_trace(
                go.Bar(
                    x=df.index,
                    y=df['MACD_Histogram'],
                    name='MACD Histogram',
                    marker_color='rgba(0,255,0,0.5)'
                ),
                row=4, col=1
            )
    
    # Volatility Bands
    if 'BB_Width' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['BB_Width'],
                name='BB Width',
                line=dict(color='orange', width=1)
            ),
            row=5, col=1
        )
    
    if 'ATR' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['ATR'],
                name='ATR',
                line=dict(color='purple', width=1)
            ),
            row=5, col=1
        )
    
    # Update layout
    fig.update_layout(
        title=f'{symbol} Technical Analysis',
        yaxis_title='Price',
        yaxis2_title='Volume',
        yaxis3_title='Momentum',
        yaxis4_title='Trend',
        yaxis5_title='Volatility',
        xaxis_rangeslider_visible=False,
        height=1000,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Add pattern annotations
    if 'Double_Top' in df.columns:
        double_tops = df[df['Double_Top']]
        for idx in double_tops.index:
            fig.add_annotation(
                x=idx,
                y=df.loc[idx, 'High'],
                text="Double Top",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor="red",
                row=1,
                col=1
            )
    
    if 'Double_Bottom' in df.columns:
        double_bottoms = df[df['Double_Bottom']]
        for idx in double_bottoms.index:
            fig.add_annotation(
                x=idx,
                y=df.loc[idx, 'Low'],
                text="Double Bottom",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor="green",
                row=1,
                col=1
            )
    
    if 'Head_Shoulders' in df.columns:
        head_shoulders = df[df['Head_Shoulders']]
        for idx in head_shoulders.index:
            fig.add_annotation(
                x=idx,
                y=df.loc[idx, 'High'],
                text="Head & Shoulders",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor="red",
                row=1,
                col=1
            )
    
    return fig

def create_performance_comparison_chart(portfolio_returns: pd.Series, 
                                       benchmark_returns: pd.Series = None) -> go.Figure:
    """Create performance comparison chart"""
    fig = go.Figure()
    
    # Portfolio cumulative returns
    portfolio_cumulative = (1 + portfolio_returns).cumprod()
    
    fig.add_trace(
        go.Scatter(
            x=portfolio_cumulative.index,
            y=portfolio_cumulative.values,
            name='Portfolio',
            line=dict(color='#2E86AB', width=3)
        )
    )
    
    # Benchmark if provided
    if benchmark_returns is not None:
        benchmark_cumulative = (1 + benchmark_returns).cumprod()
        fig.add_trace(
            go.Scatter(
                x=benchmark_cumulative.index,
                y=benchmark_cumulative.values,
                name='Benchmark',
                line=dict(color='#FF6B6B', width=2, dash='dash')
            )
        )
    
    fig.update_layout(
        title='Portfolio vs Benchmark Performance',
        xaxis_title='Date',
        yaxis_title='Cumulative Return',
        template=CHART_TEMPLATE,
        height=CHART_HEIGHT,
        hovermode='x unified'
    )
    
    return fig

def create_drawdown_chart(returns: pd.Series) -> go.Figure:
    """Create drawdown chart"""
    cumulative_returns = (1 + returns).cumprod()
    running_max = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns - running_max) / running_max
    
    fig = go.Figure()
    
    fig.add_trace(
        go.Scatter(
            x=drawdown.index,
            y=drawdown.values * 100,
            fill='tonexty',
            name='Drawdown',
            line=dict(color='red', width=2),
            fillcolor='rgba(255,0,0,0.3)'
        )
    )
    
    fig.update_layout(
        title='Portfolio Drawdown',
        xaxis_title='Date',
        yaxis_title='Drawdown (%)',
        template=CHART_TEMPLATE,
        height=CHART_HEIGHT,
        hovermode='x unified'
    )
    
    return fig

def create_risk_metrics_chart(risk_metrics: Dict) -> go.Figure:
    """Create risk metrics radar chart"""
    categories = list(risk_metrics.keys())
    values = list(risk_metrics.values())
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Risk Metrics',
        line_color='#2E86AB'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )),
        title='Risk Profile Radar Chart',
        template=CHART_TEMPLATE,
        height=CHART_HEIGHT
    )
    
    return fig

def create_correlation_heatmap(correlation_matrix: pd.DataFrame) -> go.Figure:
    """Create correlation heatmap"""
    fig = go.Figure(data=go.Heatmap(
        z=correlation_matrix.values,
        x=correlation_matrix.columns,
        y=correlation_matrix.index,
        colorscale='RdYlBu',
        zmid=0,
        text=correlation_matrix.round(2).values,
        texttemplate="%{text}",
        textfont={"size": 10},
        hoverongaps=False
    ))
    
    fig.update_layout(
        title='Portfolio Correlation Matrix',
        template=CHART_TEMPLATE,
        height=CHART_HEIGHT
    )
    
    return fig

def create_signal_strength_chart(signal_data: List[Dict]) -> go.Figure:
    """Create signal strength visualization"""
    if not signal_data:
        return None
    
    symbols = [item['symbol'] for item in signal_data]
    signal_strength = [item['signal_strength'] for item in signal_data]
    momentum_score = [item['momentum_score'] for item in signal_data]
    trend_score = [item['trend_score'] for item in signal_data]
    
    fig = go.Figure()
    
    # Signal strength bars
    fig.add_trace(go.Bar(
        x=symbols,
        y=signal_strength,
        name='Overall Signal Strength',
        marker_color='#2E86AB',
        text=[f'{x:.1f}' for x in signal_strength],
        textposition='auto'
    ))
    
    # Add momentum and trend as scatter points
    fig.add_trace(go.Scatter(
        x=symbols,
        y=momentum_score,
        mode='markers',
        name='Momentum Score',
        marker=dict(color='red', size=10, symbol='circle')
    ))
    
    fig.add_trace(go.Scatter(
        x=symbols,
        y=trend_score,
        mode='markers',
        name='Trend Score',
        marker=dict(color='green', size=10, symbol='square')
    ))
    
    fig.update_layout(
        title='Signal Strength Analysis',
        xaxis_title='Symbols',
        yaxis_title='Score (0-100)',
        template=CHART_TEMPLATE,
        height=CHART_HEIGHT,
        barmode='group'
    )
    
    return fig

def create_trade_timeline_chart(trades_df: pd.DataFrame) -> go.Figure:
    """Create trade timeline chart"""
    if trades_df.empty:
        return None
    
    fig = go.Figure()
    
    # Buy trades
    buy_trades = trades_df[trades_df['side'].str.upper() == 'BUY']
    if not buy_trades.empty:
        fig.add_trace(go.Scatter(
            x=buy_trades['trade_date'],
            y=buy_trades['avg_price'],
            mode='markers',
            name='Buy Orders',
            marker=dict(color='green', size=10, symbol='triangle-up'),
            text=buy_trades['ticker'],
            hovertemplate='<b>%{text}</b><br>Date: %{x}<br>Price: $%{y:.2f}<extra></extra>'
        ))
    
    # Sell trades
    sell_trades = trades_df[trades_df['side'].str.upper() == 'SELL']
    if not sell_trades.empty:
        fig.add_trace(go.Scatter(
            x=sell_trades['trade_date'],
            y=sell_trades['avg_price'],
            mode='markers',
            name='Sell Orders',
            marker=dict(color='red', size=10, symbol='triangle-down'),
            text=sell_trades['ticker'],
            hovertemplate='<b>%{text}</b><br>Date: %{x}<br>Price: $%{y:.2f}<extra></extra>'
        ))
    
    fig.update_layout(
        title='Trading Timeline',
        xaxis_title='Date',
        yaxis_title='Price ($)',
        template=CHART_TEMPLATE,
        height=CHART_HEIGHT,
        hovermode='closest'
    )
    
    return fig

def create_risk_return_scatter(portfolio_data: Dict) -> go.Figure:
    """Create risk-return scatter plot"""
    if not portfolio_data:
        return None
    
    symbols = []
    returns = []
    risks = []
    sizes = []
    
    for symbol, data in portfolio_data.items():
        symbols.append(symbol)
        returns.append(data.get('return_pct', 0))
        risks.append(data.get('volatility', 0))
        sizes.append(data.get('market_value', 0))
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=risks,
        y=returns,
        mode='markers+text',
        text=symbols,
        textposition='top center',
        marker=dict(
            size=[s/max(sizes)*50 + 10 for s in sizes],  # Scale marker size
            color=returns,
            colorscale='RdYlGn',
            showscale=True,
            colorbar=dict(title="Return %")
        ),
        hovertemplate='<b>%{text}</b><br>Risk: %{x:.2f}%<br>Return: %{y:.2f}%<extra></extra>'
    ))
    
    fig.update_layout(
        title='Risk-Return Analysis',
        xaxis_title='Risk (Volatility %)',
        yaxis_title='Return %',
        template=CHART_TEMPLATE,
        height=CHART_HEIGHT
    )
    
    return fig

def create_ml_feature_importance_chart(feature_importance: Dict) -> go.Figure:
    """Create feature importance chart"""
    if not feature_importance:
        return None
    
    features = list(feature_importance.keys())
    importance = list(feature_importance.values())
    
    # Sort by importance
    sorted_data = sorted(zip(features, importance), key=lambda x: x[1], reverse=True)
    features, importance = zip(*sorted_data)
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=list(importance),
        y=list(features),
        orientation='h',
        marker_color='#2E86AB',
        text=[f'{x:.3f}' for x in importance],
        textposition='auto'
    ))
    
    fig.update_layout(
        title='ML Model Feature Importance',
        xaxis_title='Importance Score',
        yaxis_title='Features',
        template=CHART_TEMPLATE,
        height=max(400, len(features) * 25),  # Dynamic height based on number of features
        margin=dict(l=150)  # More space for feature names
    )
    
    return fig 