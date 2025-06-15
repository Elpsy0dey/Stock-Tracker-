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
    """Create simplified technical analysis chart with explanatory trend lines"""
    if df.empty:
        return None
    
    # Create subplots - reduced from 5 to 3 panels for simplicity
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.10,  # Increased spacing between subplots
        row_heights=[0.55, 0.2, 0.25],  # Adjusted height ratios
        subplot_titles=(
            f'{symbol} - Price & Key Trend Lines',
            'Volume & OBV',
            'Momentum Indicators'
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
            name='Price',
            legendgroup='price',
            legendrank=1  # Control legend order
        ),
        row=1, col=1
    )
    
    # Bollinger Bands with clearer labels
    if 'BB_Upper' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['BB_Upper'],
                name='BB Upper',
                line=dict(color='rgba(250,128,114,0.7)', width=1.5, dash='dot'),
                showlegend=True,
                legendgroup='bands'
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['BB_Lower'],
                name='BB Lower',
                line=dict(color='rgba(144,238,144,0.7)', width=1.5, dash='dot'),
                fill='tonexty',
                fillcolor='rgba(235,235,235,0.2)',
                showlegend=True,
                legendgroup='bands'
            ),
            row=1, col=1
        )
    
    # Moving Averages with clearer labels
    if 'SMA_20' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['SMA_20'],
                name='SMA 20',
                line=dict(color='blue', width=1.5),
                legendgroup='moving_averages'
            ),
            row=1, col=1
        )
    
    if 'SMA_50' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['SMA_50'],
                name='SMA 50',
                line=dict(color='orange', width=1.5),
                legendgroup='moving_averages'
            ),
            row=1, col=1
        )
    
    if 'SMA_200' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['SMA_200'],
                name='SMA 200',
                line=dict(color='red', width=1.5),
                legendgroup='moving_averages'
            ),
            row=1, col=1
        )
    
    # Add trend lines and key level annotations
    if len(df) > 30:
        # Find significant highs and lows for trend lines
        recent_df = df.iloc[-30:]
        max_point_idx = recent_df['High'].idxmax()
        min_point_idx = recent_df['Low'].idxmin()
        
        # Add resistance trend line if we have a significant high
        if max_point_idx:
            fig.add_shape(
                type="line",
                x0=max_point_idx,
                y0=df.loc[max_point_idx, 'High'],
                x1=df.index[-1],
                y1=df.loc[max_point_idx, 'High'],
                line=dict(color="red", width=2, dash="dash"),
                row=1, col=1
            )
            fig.add_annotation(
                x=df.index[-5],
                y=df.loc[max_point_idx, 'High'],
                text="Resistance",
                showarrow=False,
                font=dict(color="red", size=12),
                row=1, col=1
            )
        
        # Add support trend line if we have a significant low
        if min_point_idx:
            fig.add_shape(
                type="line",
                x0=min_point_idx,
                y0=df.loc[min_point_idx, 'Low'],
                x1=df.index[-1],
                y1=df.loc[min_point_idx, 'Low'],
                line=dict(color="green", width=2, dash="dash"),
                row=1, col=1
            )
            fig.add_annotation(
                x=df.index[-5],
                y=df.loc[min_point_idx, 'Low'],
                text="Support",
                showarrow=False,
                font=dict(color="green", size=12),
                row=1, col=1
            )
    
    # Add crossover annotations
    if 'SMA_50' in df.columns and 'SMA_200' in df.columns:
        for i in range(1, len(df)):
            # Golden Cross (SMA50 crosses above SMA200)
            if df['SMA_50'].iloc[i-1] <= df['SMA_200'].iloc[i-1] and df['SMA_50'].iloc[i] > df['SMA_200'].iloc[i]:
                fig.add_annotation(
                    x=df.index[i],
                    y=df['SMA_50'].iloc[i],
                    text="Golden Cross",
                    showarrow=True,
                    arrowhead=2,
                    arrowcolor="green",
                    font=dict(color="green", size=12),
                    row=1, col=1
                )
            
            # Death Cross (SMA50 crosses below SMA200)
            if df['SMA_50'].iloc[i-1] >= df['SMA_200'].iloc[i-1] and df['SMA_50'].iloc[i] < df['SMA_200'].iloc[i]:
                fig.add_annotation(
                    x=df.index[i],
                    y=df['SMA_50'].iloc[i],
                    text="Death Cross",
                    showarrow=True,
                    arrowhead=2,
                    arrowcolor="red",
                    font=dict(color="red", size=12),
                    row=1, col=1
                )
    
    # Volume and OBV with clearer explanation
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df['Volume'],
            name='Volume',
            marker_color='rgba(0,0,255,0.5)',
            legendgroup='volume'
        ),
        row=2, col=1
    )
    
    if 'OBV' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['OBV'],
                name='OBV',
                line=dict(color='green', width=1.5),
                legendgroup='volume'
            ),
            row=2, col=1
        )
        
        # Add OBV trend line
        obv_trend = df['OBV'].rolling(window=10).mean()
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=obv_trend,
                name='OBV Trend',
                line=dict(color='purple', width=1, dash='dash'),
                showlegend=False
            ),
            row=2, col=1
        )
    
    # Combined Momentum Indicators (RSI + MACD)
    if 'RSI' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['RSI'],
                name='RSI',
                line=dict(color='purple', width=1.5),
                legendgroup='momentum'
            ),
            row=3, col=1
        )
        
        # Add RSI levels with explanations
        fig.add_shape(
            type="line",
            x0=df.index[0],
            y0=70,
            x1=df.index[-1],
            y1=70,
            line=dict(color="red", width=1, dash="dash"),
            row=3, col=1
        )
        fig.add_annotation(
            x=df.index[0],
            y=70,
            text="Overbought",
            showarrow=False,
            font=dict(color="red", size=10),
            xanchor="left",
            row=3, col=1
        )
        
        fig.add_shape(
            type="line",
            x0=df.index[0],
            y0=30,
            x1=df.index[-1],
            y1=30,
            line=dict(color="green", width=1, dash="dash"),
            row=3, col=1
        )
        fig.add_annotation(
            x=df.index[0],
            y=30,
            text="Oversold",
            showarrow=False,
            font=dict(color="green", size=10),
            xanchor="left",
            row=3, col=1
        )
    
    # Add MACD
    if 'MACD' in df.columns and 'MACD_Signal' in df.columns:
        # Create a secondary y-axis for MACD
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['MACD'],
                name='MACD',
                line=dict(color='blue', width=1.5),
                yaxis="y4",
                legendgroup='momentum'
            ),
            row=3, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['MACD_Signal'],
                name='Signal',
                line=dict(color='red', width=1.5),
                yaxis="y4",
                legendgroup='momentum'
            ),
            row=3, col=1
        )
        
        # Add MACD crossover annotations
        for i in range(5, len(df)):
            # MACD crosses above signal (bullish)
            if df['MACD'].iloc[i-1] <= df['MACD_Signal'].iloc[i-1] and df['MACD'].iloc[i] > df['MACD_Signal'].iloc[i]:
                fig.add_annotation(
                    x=df.index[i],
                    y=df['MACD'].iloc[i],
                    text="Buy",
                    showarrow=True,
                    arrowhead=2,
                    arrowcolor="green",
                    font=dict(color="green", size=10),
                    yshift=10,
                    row=3, col=1
                )
            
            # MACD crosses below signal (bearish)
            if df['MACD'].iloc[i-1] >= df['MACD_Signal'].iloc[i-1] and df['MACD'].iloc[i] < df['MACD_Signal'].iloc[i]:
                fig.add_annotation(
                    x=df.index[i],
                    y=df['MACD'].iloc[i],
                    text="Sell",
                    showarrow=True,
                    arrowhead=2,
                    arrowcolor="red",
                    font=dict(color="red", size=10),
                    yshift=-10,
                    row=3, col=1
                )
    
    # Update layout
    fig.update_layout(
        title=f'{symbol} Technical Analysis with Trend Lines',
        yaxis_title='Price',
        yaxis2_title='Volume',
        yaxis3_title='RSI',
        xaxis_rangeslider_visible=False,
        height=850,  # Increased height for more spacing
        width=1200,  # Make chart wider
        showlegend=True,
        # Add a secondary y-axis for MACD
        yaxis4=dict(
            title="MACD",
            anchor="x",
            overlaying="y3",
            side="right"
        ),
        margin=dict(b=180),  # Further increase bottom margin for legend
        # Add time axis for each subplot
        xaxis=dict(
            showticklabels=True,
            tickformat="%b %d",
            dtick="M2",  # Bi-monthly ticks to reduce crowding
            tickangle=45
        ),
        xaxis2=dict(
            showticklabels=True,
            tickformat="%b %d",
            dtick="M2",  # Bi-monthly ticks
            tickangle=0  # Flat labels
        ),
        xaxis3=dict(
            showticklabels=True,
            tickformat="%b %d",
            dtick="M2",  # Bi-monthly ticks
            tickangle=0  # Flat labels
        )
    )
    
    # Create legend layout similar to the second image
    fig.update_layout(
        legend=dict(
            groupclick="toggleitem",
            itemsizing="constant",
            itemwidth=40,  # Narrower legend items
            traceorder="normal",
            tracegroupgap=5,  # Tighter spacing
            orientation="h",
            xanchor="center",
            x=0.5,
            yanchor="bottom",
            y=-0.30,
            bgcolor="rgba(0,0,0,0.8)",
            bordercolor="DarkSlateGrey",
            borderwidth=1,
            font=dict(size=10, color="white")
        )
    )
    
    # Add explanatory note at the bottom of the chart
    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.5, y=-0.22,
        text="Moving Averages (trend), Volume (buying/selling pressure), MACD & Squeeze Momentum (momentum)",
        showarrow=False,
        font=dict(size=12, color="white"),
        align="center",
        bgcolor="rgba(0,0,0,0.8)",
        bordercolor="DarkSlateGrey",
        borderwidth=1
    )
    
    # Update legend with title
    fig.update_layout(
        legend_title_text="Key Trend Lines",
        legend_title_font=dict(size=12, color="white")
    )
    
    # Organize legend items by category
    for trace in fig.data:
        if trace.name == 'Price':
            trace.update(legendrank=1)
        elif 'BB' in trace.name:
            trace.update(legendrank=2)
        elif 'SMA' in trace.name:
            trace.update(legendrank=3)
        elif trace.name == 'Volume':
            trace.update(legendrank=4)
        elif trace.name == 'OBV':
            trace.update(legendrank=5)
        elif trace.name == 'RSI':
            trace.update(legendrank=6)
        elif trace.name == 'MACD':
            trace.update(legendrank=7)
        elif trace.name == 'Signal':
            trace.update(legendrank=8)
    
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

def create_explanatory_chart(df: pd.DataFrame, symbol: str) -> go.Figure:
    """Create a simplified chart with educational annotations explaining indicators"""
    if df.empty or len(df) < 30:
        return None
    
    # Use only the last 90 days of data for clarity
    if len(df) > 90:
        df = df.iloc[-90:]
    
    # Create a simple figure with 2 rows
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        row_heights=[0.7, 0.3],
        subplot_titles=(
            f'{symbol} - Price Action Explained',
            'Key Momentum Indicator'
        )
    )
    
    # Add price candlesticks
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='Price',
            increasing_line_color='green',
            decreasing_line_color='red'
        ),
        row=1, col=1
    )
    
    # Add simple moving averages
    if 'SMA_20' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['SMA_20'],
                name='20-day Trend',
                line=dict(color='blue', width=2)
            ),
            row=1, col=1
        )
    
    if 'SMA_50' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['SMA_50'],
                name='50-day Trend',
                line=dict(color='orange', width=2)
            ),
            row=1, col=1
        )
    
    # Add RSI with clearer explanation
    if 'RSI' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['RSI'],
                name='RSI (Momentum)',
                line=dict(color='purple', width=2)
            ),
            row=2, col=1
        )
        
        # Add overbought/oversold zones with fill
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=[70] * len(df),
                name='Overbought',
                line=dict(color='rgba(255,0,0,0.5)', width=1, dash='dash'),
                fill=None
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=[30] * len(df),
                name='Oversold',
                line=dict(color='rgba(0,255,0,0.5)', width=1, dash='dash'),
                fill='tonexty',
                fillcolor='rgba(0,255,0,0.1)'
            ),
            row=2, col=1
        )
    
    # Find key patterns and add annotations
    # 1. Find a significant high point
    if len(df) >= 30:
        high_point_idx = df['High'].idxmax()
        low_point_idx = df['Low'].idxmin()
        
        # Annotate high point
        fig.add_annotation(
            x=high_point_idx,
            y=df.loc[high_point_idx, 'High'],
            text="Resistance Level",
            showarrow=True,
            arrowhead=2,
            arrowcolor="red",
            font=dict(color="red", size=12),
            row=1, col=1
        )
        
        # Annotate low point
        fig.add_annotation(
            x=low_point_idx,
            y=df.loc[low_point_idx, 'Low'],
            text="Support Level",
            showarrow=True,
            arrowhead=2,
            arrowcolor="green",
            font=dict(color="green", size=12),
            row=1, col=1
        )
        
        # Add trend lines
        fig.add_shape(
            type="line",
            x0=low_point_idx,
            y0=df.loc[low_point_idx, 'Low'],
            x1=df.index[-1],
            y1=df.loc[low_point_idx, 'Low'],
            line=dict(color="green", width=2, dash="dash"),
            row=1, col=1
        )
        
        fig.add_shape(
            type="line",
            x0=high_point_idx,
            y0=df.loc[high_point_idx, 'High'],
            x1=df.index[-1],
            y1=df.loc[high_point_idx, 'High'],
            line=dict(color="red", width=2, dash="dash"),
            row=1, col=1
        )
    
    # Find and annotate trend changes
    if 'SMA_20' in df.columns and 'SMA_50' in df.columns:
        # Find crossovers
        for i in range(5, len(df)):
            # Bullish crossover
            if (df['SMA_20'].iloc[i-1] <= df['SMA_50'].iloc[i-1] and 
                df['SMA_20'].iloc[i] > df['SMA_50'].iloc[i]):
                fig.add_annotation(
                    x=df.index[i],
                    y=df['SMA_20'].iloc[i],
                    text="Bullish Trend Change",
                    showarrow=True,
                    arrowhead=2,
                    arrowcolor="green",
                    font=dict(color="green", size=12),
                    row=1, col=1
                )
            
            # Bearish crossover
            if (df['SMA_20'].iloc[i-1] >= df['SMA_50'].iloc[i-1] and 
                df['SMA_20'].iloc[i] < df['SMA_50'].iloc[i]):
                fig.add_annotation(
                    x=df.index[i],
                    y=df['SMA_20'].iloc[i],
                    text="Bearish Trend Change",
                    showarrow=True,
                    arrowhead=2,
                    arrowcolor="red",
                    font=dict(color="red", size=12),
                    row=1, col=1
                )
    
    # Find and annotate RSI signals
    if 'RSI' in df.columns:
        # Find oversold conditions followed by crossing above 30
        for i in range(5, len(df)):
            if df['RSI'].iloc[i-1] < 30 and df['RSI'].iloc[i] > 30:
                fig.add_annotation(
                    x=df.index[i],
                    y=35,  # Just above the oversold line
                    text="Buy Signal",
                    showarrow=True,
                    arrowhead=2,
                    arrowcolor="green",
                    font=dict(color="green", size=12),
                    row=2, col=1
                )
            
            # Find overbought conditions followed by crossing below 70
            if df['RSI'].iloc[i-1] > 70 and df['RSI'].iloc[i] < 70:
                fig.add_annotation(
                    x=df.index[i],
                    y=65,  # Just below the overbought line
                    text="Sell Signal",
                    showarrow=True,
                    arrowhead=2,
                    arrowcolor="red",
                    font=dict(color="red", size=12),
                    row=2, col=1
                )
    
    # Add educational annotations explaining the chart elements
    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.01, y=0.99,
        text="Green candles = Price up, Red candles = Price down",
        showarrow=False,
        font=dict(size=10),
        align="left",
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="black",
        borderwidth=1
    )
    
    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.01, y=0.95,
        text="Blue line = Short-term trend (20 days)",
        showarrow=False,
        font=dict(size=10),
        align="left",
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="black",
        borderwidth=1
    )
    
    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.01, y=0.91,
        text="Orange line = Medium-term trend (50 days)",
        showarrow=False,
        font=dict(size=10),
        align="left",
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="black",
        borderwidth=1
    )
    
    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.01, y=0.25,
        text="RSI > 70 = Overbought (potential sell)",
        showarrow=False,
        font=dict(size=10),
        align="left",
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="black",
        borderwidth=1
    )
    
    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.01, y=0.21,
        text="RSI < 30 = Oversold (potential buy)",
        showarrow=False,
        font=dict(size=10),
        align="left",
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="black",
        borderwidth=1
    )
    
    # Update layout
    fig.update_layout(
        title="Stock Chart Explained: How to Read Technical Indicators",
        yaxis_title="Price ($)",
        yaxis2_title="RSI (0-100)",
        xaxis_rangeslider_visible=False,
        height=700,
        template="plotly_white",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.25,  # Position below the chart
            xanchor="center",
            x=0.5,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="LightGrey",
            borderwidth=1,
            font=dict(size=10),
            itemsizing="constant",
            tracegroupgap=5
        ),
        margin=dict(b=100)  # Add bottom margin for legend
    )
    
    # Create grouped legends with custom categories and improve spacing
    fig.update_layout(
        legend=dict(
            groupclick="toggleitem",
            itemsizing="constant",
            itemwidth=50,  # Narrower legend items
            traceorder="normal",
            tracegroupgap=10,  # Less space between legend items
            orientation="h",
            xanchor="center",
            x=0.5,
            yanchor="bottom",
            y=-0.30,
            bgcolor="rgba(0,0,0,0.8)",
            bordercolor="DarkSlateGrey",
            borderwidth=1,
            font=dict(size=10, color="white")
        )
    )
    
    # Set y-axis range for RSI
    fig.update_yaxes(range=[0, 100], row=2, col=1)
    
    return fig

def create_combined_indicators_chart(df: pd.DataFrame, symbol: str) -> go.Figure:
    """Create a chart with both MACD Ultimate and Squeeze Momentum indicators"""
    if df.empty:
        return None
    
    # Create subplots with 4 rows
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.10,  # Increased spacing between subplots
        row_heights=[0.45, 0.18, 0.18, 0.19],  # Adjusted height ratios
        subplot_titles=(
            f'{symbol} - Price Action',
            'Volume',
            'MACD Ultimate',
            'Squeeze Momentum'
        )
    )
    
    # Main price chart
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='Price',
            legendgroup='price'
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
                line=dict(color='blue', width=1.5),
                legendgroup='moving_averages'
            ),
            row=1, col=1
        )
    
    if 'SMA_50' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['SMA_50'],
                name='SMA 50',
                line=dict(color='orange', width=1.5),
                legendgroup='moving_averages'
            ),
            row=1, col=1
        )
    
    # Volume
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df['Volume'],
            name='Volume',
            marker_color='rgba(0,0,255,0.5)',
            legendgroup='volume'
        ),
        row=2, col=1
    )
    
    # Calculate MACD if not already in the dataframe
    if not all(col in df.columns for col in ['MACD', 'MACD_Signal', 'MACD_Histogram']):
        # Use the values from the technical analyzer settings
        fast_length = 12
        slow_length = 26
        signal_length = 9
        
        # Calculate MACD components
        fast_ema = df['Close'].ewm(span=fast_length).mean()
        slow_ema = df['Close'].ewm(span=slow_length).mean()
        macd = fast_ema - slow_ema
        signal = macd.rolling(window=signal_length).mean()
        histogram = macd - signal
        
        # Store in dataframe
        df['MACD'] = macd
        df['MACD_Signal'] = signal
        df['MACD_Histogram'] = histogram
    
    # MACD Ultimate (similar to TradingView's CM_MacD_Ult_MTF)
    if 'MACD' in df.columns and 'MACD_Signal' in df.columns and 'MACD_Histogram' in df.columns:
        # Add MACD line
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['MACD'],
                name='MACD',
                line=dict(color='blue', width=2),
                legendgroup='macd'
            ),
            row=3, col=1
        )
        
        # Add Signal line
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['MACD_Signal'],
                name='Signal',
                line=dict(color='red', width=1.5),
                legendgroup='macd'
            ),
            row=3, col=1
        )
        
        # Add Histogram with 4 colors based on direction and zero line
        colors = []
        for i in range(len(df)):
            if i == 0:
                colors.append('rgba(0,255,0,0.5)')  # Default color
                continue
                
            # Above zero and rising
            if df['MACD_Histogram'].iloc[i] > 0 and df['MACD_Histogram'].iloc[i] > df['MACD_Histogram'].iloc[i-1]:
                colors.append('rgba(0,255,255,0.7)')  # Aqua
            # Above zero and falling
            elif df['MACD_Histogram'].iloc[i] > 0 and df['MACD_Histogram'].iloc[i] <= df['MACD_Histogram'].iloc[i-1]:
                colors.append('rgba(0,0,255,0.7)')  # Blue
            # Below zero and falling
            elif df['MACD_Histogram'].iloc[i] <= 0 and df['MACD_Histogram'].iloc[i] <= df['MACD_Histogram'].iloc[i-1]:
                colors.append('rgba(255,0,0,0.7)')  # Red
            # Below zero and rising
            elif df['MACD_Histogram'].iloc[i] <= 0 and df['MACD_Histogram'].iloc[i] > df['MACD_Histogram'].iloc[i-1]:
                colors.append('rgba(128,0,0,0.7)')  # Maroon
            else:
                colors.append('rgba(0,255,0,0.5)')  # Default
        
        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df['MACD_Histogram'],
                name='Histogram',
                marker_color=colors,
                legendgroup='macd',
                showlegend=False
            ),
            row=3, col=1
        )
        
        # Add zero line
        fig.add_shape(
            type="line",
            x0=df.index[0],
            y0=0,
            x1=df.index[-1],
            y1=0,
            line=dict(color="white", width=1),
            row=3, col=1
        )
        
        # Add dots at crossover points
        for i in range(1, len(df)):
            # MACD crosses above signal
            if df['MACD'].iloc[i-1] <= df['MACD_Signal'].iloc[i-1] and df['MACD'].iloc[i] > df['MACD_Signal'].iloc[i]:
                fig.add_trace(
                    go.Scatter(
                        x=[df.index[i]],
                        y=[df['MACD'].iloc[i]],
                        mode='markers',
                        marker=dict(color='lime', size=8),
                        name='Buy Signal',
                        legendgroup='macd_signals',
                        showlegend=(i == 1)  # Show only once in legend
                    ),
                    row=3, col=1
                )
            
            # MACD crosses below signal
            if df['MACD'].iloc[i-1] >= df['MACD_Signal'].iloc[i-1] and df['MACD'].iloc[i] < df['MACD_Signal'].iloc[i]:
                fig.add_trace(
                    go.Scatter(
                        x=[df.index[i]],
                        y=[df['MACD'].iloc[i]],
                        mode='markers',
                        marker=dict(color='red', size=8),
                        name='Sell Signal',
                        legendgroup='macd_signals',
                        showlegend=(i == 1)  # Show only once in legend
                    ),
                    row=3, col=1
                )
    
    # Calculate Squeeze Momentum if not in dataframe
    # Implementing LazyBear's Squeeze Momentum indicator
    if not all(col in df.columns for col in ['SQZ_ON', 'SQZ_OFF', 'SQZ_NOSC', 'SQZ_VAL']):
        # Parameters
        length_bb = 20
        mult_bb = 2.0
        length_kc = 20
        mult_kc = 1.5
        
        # Calculate Bollinger Bands
        basis = df['Close'].rolling(window=length_bb).mean()
        dev = mult_bb * df['Close'].rolling(window=length_bb).std()
        upper_bb = basis + dev
        lower_bb = basis - dev
        
        # Calculate Keltner Channels
        ma = df['Close'].rolling(window=length_kc).mean()
        range_val = df['High'] - df['Low']  # Using high-low range
        range_ma = range_val.rolling(window=length_kc).mean()
        upper_kc = ma + range_ma * mult_kc
        lower_kc = ma - range_ma * mult_kc
        
        # Determine squeeze state
        sqz_on = (lower_bb > lower_kc) & (upper_bb < upper_kc)
        sqz_off = (lower_bb < lower_kc) & (upper_bb > upper_kc)
        no_sqz = (~sqz_on) & (~sqz_off)
        
        # Calculate momentum value
        highest_high = df['High'].rolling(window=length_kc).max()
        lowest_low = df['Low'].rolling(window=length_kc).min()
        avg_val = ((highest_high + lowest_low) / 2 + ma) / 2
        
        # Linear regression of close price deviation from average
        # This is simplified - in a real implementation you'd use proper linear regression
        val = df['Close'] - avg_val
        val_smooth = val.rolling(window=length_kc).mean()
        
        # Store in dataframe
        df['SQZ_ON'] = sqz_on
        df['SQZ_OFF'] = sqz_off
        df['SQZ_NOSC'] = no_sqz
        df['SQZ_VAL'] = val_smooth
    
    # Squeeze Momentum (LazyBear's indicator)
    if all(col in df.columns for col in ['SQZ_ON', 'SQZ_OFF', 'SQZ_NOSC', 'SQZ_VAL']):
        # Determine colors for histogram bars
        sqz_colors = []
        for i in range(len(df)):
            if i == 0:
                sqz_colors.append('gray')
                continue
                
            val = df['SQZ_VAL'].iloc[i]
            prev_val = df['SQZ_VAL'].iloc[i-1]
            
            if val > 0:
                if val > prev_val:
                    sqz_colors.append('lime')  # Positive and increasing
                else:
                    sqz_colors.append('green')  # Positive and decreasing
            else:
                if val < prev_val:
                    sqz_colors.append('red')  # Negative and decreasing
                else:
                    sqz_colors.append('maroon')  # Negative and increasing
        
        # Add histogram for momentum value
        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df['SQZ_VAL'],
                name='Squeeze Momentum',
                marker_color=sqz_colors,
                legendgroup='squeeze'
            ),
            row=4, col=1
        )
        
        # Add dots for squeeze state
        squeeze_dots_colors = []
        for i in range(len(df)):
            if df['SQZ_ON'].iloc[i]:
                squeeze_dots_colors.append('black')  # Squeeze is on
            elif df['SQZ_OFF'].iloc[i]:
                squeeze_dots_colors.append('gray')  # Squeeze is off
            else:
                squeeze_dots_colors.append('blue')  # No squeeze
        
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=[0] * len(df),
                mode='markers',
                marker=dict(color=squeeze_dots_colors, size=6),
                name='Squeeze State',
                legendgroup='squeeze'
            ),
            row=4, col=1
        )
        
        # Add zero line
        fig.add_shape(
            type="line",
            x0=df.index[0],
            y0=0,
            x1=df.index[-1],
            y1=0,
            line=dict(color="white", width=1),
            row=4, col=1
        )
    
    # Update layout
    fig.update_layout(
        title=f'{symbol} - Combined Technical Indicators',
        xaxis4_title='Date',
        yaxis_title='Price',
        yaxis2_title='Volume',
        yaxis3_title='MACD',
        yaxis4_title='Momentum',
        xaxis_rangeslider_visible=False,
        height=950,  # Increased height for better spacing
        width=1200,  # Make chart wider
        showlegend=True,
        margin=dict(b=180),  # Further increase bottom margin for legend
        # Add time axis for each subplot
        xaxis=dict(
            showticklabels=True,
            tickformat="%b %d",
            dtick="M2",  # Bi-monthly ticks to reduce crowding
            tickangle=45
        ),
        xaxis2=dict(
            showticklabels=True,
            tickformat="%b %d",
            dtick="M2",  # Bi-monthly ticks
            tickangle=0  # Flat labels
        ),
        xaxis3=dict(
            showticklabels=True,
            tickformat="%b %d",
            dtick="M2",  # Bi-monthly ticks
            tickangle=0  # Flat labels
        ),
        xaxis4=dict(
            showticklabels=True,
            tickformat="%b %d",
            dtick="M2",  # Bi-monthly ticks
            tickangle=0  # Flat labels
        )
    )
    
    # Add legend title and groups
    fig.update_layout(
        legend=dict(
            groupclick="toggleitem",
            itemsizing="constant",
            itemwidth=40,  # Narrower legend items
            traceorder="normal",
            tracegroupgap=5,  # Tighter spacing
            orientation="h",
            xanchor="center",
            x=0.5,
            yanchor="bottom",
            y=-0.30,
            bgcolor="rgba(0,0,0,0.8)",
            bordercolor="DarkSlateGrey",
            borderwidth=1,
            font=dict(size=10, color="white")
        )
    )
    
    # Add explanatory note at the bottom of the chart
    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.5, y=-0.22,
        text="Moving Averages (trend), Bollinger Bands (volatility), Volume & OBV (buying/selling pressure), RSI & MACD (momentum)",
        showarrow=False,
        font=dict(size=12, color="white"),
        align="center",
        bgcolor="rgba(0,0,0,0.8)",
        bordercolor="DarkSlateGrey",
        borderwidth=1
    )
    
    # Update legend with title
    fig.update_layout(
        legend_title_text="Key Trend Lines",
        legend_title_font=dict(size=12, color="white")
    )
    
    # Organize legend items by category
    for trace in fig.data:
        if trace.name == 'Price':
            trace.update(legendrank=1)
        elif 'BB' in trace.name:
            trace.update(legendrank=2)
        elif 'SMA' in trace.name:
            trace.update(legendrank=3)
        elif trace.name == 'Volume':
            trace.update(legendrank=4)
        elif trace.name == 'OBV':
            trace.update(legendrank=5)
        elif trace.name == 'RSI':
            trace.update(legendrank=6)
        elif trace.name == 'MACD':
            trace.update(legendrank=7)
        elif trace.name == 'Signal':
            trace.update(legendrank=8)
        elif trace.name == 'Squeeze Momentum':
            trace.update(legendrank=9)
        elif trace.name == 'Squeeze State':
            trace.update(legendrank=10)
        elif trace.name == 'Buy Signal':
            trace.update(legendrank=11)
    
    return fig 