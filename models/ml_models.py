"""
Machine Learning Module for Trading Portfolio Tracker

Implements ML approaches from the research study:
- Feature selection using 88+ technical indicators
- Ensemble models (Random Forest, XGBoost, LightGBM) for signal ranking
- Trend forecasting with LSTM/GRU models
- Breakout classification models
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ML imports with error handling
try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
    from sklearn.svm import SVC
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.feature_selection import SelectKBest, f_classif
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

# TensorFlow is not supported in Python 3.13
HAS_TENSORFLOW = False

from config.settings import *
from models.technical_analysis import TechnicalAnalyzer
from utils.data_utils import get_stock_data

class MLSignalPredictor:
    """Machine Learning engine for trading signal prediction"""
    
    def __init__(self):
        self.analyzer = TechnicalAnalyzer()
        self.models = {
            'ensemble': {},  # Tree-based models
            'breakout': {},  # Breakout classification models
            'trend': None    # LSTM model
        }
        self.feature_importance = {}
        self.scaler = StandardScaler() if HAS_SKLEARN else None
        self.is_trained = False
        
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare feature set from technical indicators
        Based on research study's 88 technical indicators
        """
        if df.empty or len(df) < 200:
            return pd.DataFrame()
        
        # Calculate all technical indicators
        df_with_indicators = self.analyzer.calculate_all_indicators(df)
        
        if df_with_indicators.empty:
            return pd.DataFrame()
        
        # Create additional features
        features_df = df_with_indicators.copy()
        
        # Price-based features
        features_df['Price_SMA20_Ratio'] = features_df['Close'] / features_df['SMA_20']
        features_df['Price_SMA50_Ratio'] = features_df['Close'] / features_df['SMA_50']
        features_df['Price_SMA200_Ratio'] = features_df['Close'] / features_df['SMA_200']
        
        # Volatility features
        features_df['ATR_Percent'] = features_df['ATR'] / features_df['Close']
        features_df['Price_Range'] = (features_df['High'] - features_df['Low']) / features_df['Close']
        features_df['BB_Width'] = (features_df['BB_Upper'] - features_df['BB_Lower']) / features_df['BB_Middle']
        
        # Volume features
        features_df['Volume_Ratio'] = features_df['Volume'] / features_df['Volume_SMA']
        features_df['Price_Volume'] = features_df['Close'] * features_df['Volume']
        features_df['OBV'] = (np.sign(features_df['Close'].diff()) * features_df['Volume']).fillna(0).cumsum()
        features_df['ADI'] = ((features_df['Close'] - features_df['Low']) - (features_df['High'] - features_df['Close'])) / (features_df['High'] - features_df['Low']) * features_df['Volume']
        
        # Momentum features
        features_df['RSI_SMA'] = features_df['RSI'].rolling(5).mean()
        features_df['MACD_Ratio'] = features_df['MACD'] / features_df['MACD_Signal']
        features_df['Williams_R'] = (features_df['High'].rolling(14).max() - features_df['Close']) / (features_df['High'].rolling(14).max() - features_df['Low'].rolling(14).min()) * -100
        
        # Trend features
        features_df['SMA_Trend'] = (features_df['SMA_20'] > features_df['SMA_50']).astype(int)
        features_df['Long_Trend'] = (features_df['SMA_50'] > features_df['SMA_200']).astype(int)
        features_df['ADX_Trend'] = (features_df['ADX'] > 25).astype(int)
        
        # Pattern features
        features_df['Higher_High'] = (features_df['High'] > features_df['High'].shift(1)).astype(int)
        features_df['Higher_Low'] = (features_df['Low'] > features_df['Low'].shift(1)).astype(int)
        features_df['Lower_High'] = (features_df['High'] < features_df['High'].shift(1)).astype(int)
        features_df['Lower_Low'] = (features_df['Low'] < features_df['Low'].shift(1)).astype(int)
        
        # Cross-sectional features
        rolling_window = 50
        features_df['RSI_Percentile'] = features_df['RSI'].rolling(rolling_window).rank(pct=True)
        features_df['Volume_Percentile'] = features_df['Volume'].rolling(rolling_window).rank(pct=True)
        features_df['Price_Percentile'] = features_df['Close'].rolling(rolling_window).rank(pct=True)
        
        # Return features
        for period in [1, 3, 5, 10, 20]:
            features_df[f'Return_{period}d'] = features_df['Close'].pct_change(period)
        
        # Clean NaN values
        features_df = features_df.fillna(method='ffill').fillna(method='bfill')
        
        # Remove any remaining NaN values
        features_df = features_df.replace([np.inf, -np.inf], np.nan)
        features_df = features_df.fillna(0)
        
        return features_df
    
    def create_labels(self, df: pd.DataFrame, prediction_horizon: int = 5, 
                     threshold: float = 0.02) -> pd.Series:
        """
        Create prediction labels based on future returns
        
        Args:
            df: DataFrame with price data
            prediction_horizon: Days ahead to predict
            threshold: Return threshold for positive label
            
        Returns:
            Series with binary labels (1 for positive returns, 0 for negative)
        """
        future_returns = df['Close'].pct_change(prediction_horizon).shift(-prediction_horizon)
        labels = (future_returns > threshold).astype(int)
        return labels
    
    def create_breakout_labels(self, df: pd.DataFrame, 
                             breakout_threshold: float = 0.05) -> pd.Series:
        """
        Create labels for breakout detection
        
        Args:
            df: DataFrame with price data
            breakout_threshold: Price change threshold for breakout
            
        Returns:
            Series with binary labels (1 for breakout, 0 for no breakout)
        """
        # Calculate rolling high
        rolling_high = df['High'].rolling(20).max()
        
        # Detect breakouts
        price_above_high = df['Close'] > rolling_high.shift(1)
        significant_move = df['Close'].pct_change(5) > breakout_threshold
        
        # Volume confirmation
        volume_surge = df['Volume'] > df['Volume'].rolling(20).mean() * 1.5
        
        # Combine conditions
        breakout = (price_above_high & significant_move & volume_surge).astype(int)
        
        return breakout
    
    def select_features(self, X: pd.DataFrame, y: pd.Series, 
                       max_features: int = 20) -> List[str]:
        """
        Select most important features using statistical tests
        
        Args:
            X: Feature matrix
            y: Target labels
            max_features: Maximum number of features to select
            
        Returns:
            List of selected feature names
        """
        if not HAS_SKLEARN:
            return list(X.columns[:max_features])
        
        # Remove rows with NaN values
        valid_mask = ~(X.isna().any(axis=1) | y.isna())
        X_clean = X[valid_mask]
        y_clean = y[valid_mask]
        
        if len(X_clean) < 50:
            return list(X.columns[:max_features])
        
        # Use SelectKBest for feature selection
        selector = SelectKBest(score_func=f_classif, k=min(max_features, X_clean.shape[1]))
        
        try:
            selector.fit(X_clean, y_clean)
            selected_features = X.columns[selector.get_support()].tolist()
            
            # Store feature scores
            feature_scores = dict(zip(X.columns, selector.scores_))
            self.feature_importance['statistical_scores'] = feature_scores
            
            return selected_features
        except:
            return list(X.columns[:max_features])
    
    def train_ensemble_models(self, symbols: List[str], 
                            lookback_days: int = ML_LOOKBACK_PERIOD) -> Dict[str, Any]:
        """
        Train ensemble models on multiple symbols
        
        Args:
            symbols: List of stock symbols to train on
            lookback_days: Number of days of historical data
            
        Returns:
            Training results and model performance metrics
        """
        if not HAS_SKLEARN:
            return {'error': 'scikit-learn not available'}
        
        all_features = []
        all_labels = []
        all_breakout_labels = []
        training_results = {}
        
        # Collect training data from multiple symbols
        for symbol in symbols[:50]:  # Limit for performance
            try:
                df = get_stock_data(symbol, period="2y", interval="1d")
                if df.empty or len(df) < 200:
                    continue
                
                # Prepare features and labels
                features_df = self.prepare_features(df)
                if features_df.empty:
                    continue
                
                # Create labels for different tasks
                return_labels = self.create_labels(features_df)
                breakout_labels = self.create_breakout_labels(features_df)
                
                # Select only numeric columns for features
                numeric_features = features_df.select_dtypes(include=[np.number]).columns
                feature_data = features_df[numeric_features].iloc[:-5]  # Remove last 5 rows (no labels)
                return_label_data = return_labels.iloc[:-5]
                breakout_label_data = breakout_labels.iloc[:-5]
                
                # Remove rows with NaN
                valid_mask = ~(feature_data.isna().any(axis=1) | return_label_data.isna() | breakout_label_data.isna())
                if valid_mask.sum() < 50:
                    continue
                
                all_features.append(feature_data[valid_mask])
                all_labels.append(return_label_data[valid_mask])
                all_breakout_labels.append(breakout_label_data[valid_mask])
                
            except Exception as e:
                print(f"Error processing {symbol}: {e}")
                continue
        
        if not all_features:
            return {'error': 'No valid training data'}
        
        # Combine all data
        X = pd.concat(all_features, ignore_index=True)
        y_return = pd.concat(all_labels, ignore_index=True)
        y_breakout = pd.concat(all_breakout_labels, ignore_index=True)
        
        # Feature selection
        selected_features = self.select_features(X, y_return)
        X_selected = X[selected_features]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_selected, y_return, test_size=1-ML_TRAIN_TEST_SPLIT, 
            random_state=ML_RANDOM_STATE, stratify=y_return
        )
        
        # Scale features
        if self.scaler:
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
        else:
            X_train_scaled = X_train
            X_test_scaled = X_test
        
        # Train ensemble models
        models_to_train = {
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=ML_RANDOM_STATE),
            'gradient_boosting': GradientBoostingClassifier(n_estimators=100, random_state=ML_RANDOM_STATE),
            'adaboost': AdaBoostClassifier(n_estimators=100, random_state=ML_RANDOM_STATE)
        }
        
        if HAS_XGBOOST:
            models_to_train['xgboost'] = xgb.XGBClassifier(n_estimators=100, random_state=ML_RANDOM_STATE)
        
        if HAS_LIGHTGBM:
            models_to_train['lightgbm'] = lgb.LGBMClassifier(n_estimators=100, random_state=ML_RANDOM_STATE)
        
        # Train breakout classification models
        breakout_models = {
            'svm': SVC(probability=True, random_state=ML_RANDOM_STATE),
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=ML_RANDOM_STATE)
        }
        
        model_results = {}
        
        # Train ensemble models
        for model_name, model in models_to_train.items():
            try:
                # Train model
                model.fit(X_train_scaled, y_train)
                
                # Make predictions
                y_pred = model.predict(X_test_scaled)
                y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
                
                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, zero_division=0)
                recall = recall_score(y_test, y_pred, zero_division=0)
                f1 = f1_score(y_test, y_pred, zero_division=0)
                
                # Cross-validation score
                cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
                
                # Store metrics
                metrics = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std()
                }
                
                # Get feature importance
                if hasattr(model, 'feature_importances_'):
                    importance_dict = dict(zip(selected_features, model.feature_importances_))
                    metrics['feature_importance'] = importance_dict
                
                # Store results
                model_results[model_name] = metrics
                self.models['ensemble'][model_name] = model
                self.feature_importance[model_name] = metrics
                
            except Exception as e:
                print(f"Error training {model_name}: {e}")
                continue
        
        # Train breakout models
        for model_name, model in breakout_models.items():
            try:
                # Train model
                model.fit(X_train_scaled, y_breakout.iloc[:len(X_train_scaled)])
                
                # Store the model
                self.models['breakout'][model_name] = model
                
            except Exception as e:
                print(f"Error training breakout model {model_name}: {e}")
                continue
        
        # Select best model based on cross-validation score
        if model_results:
            best_model_name = max(model_results.keys(), key=lambda x: model_results[x]['cv_mean'])
            self.best_model = self.models['ensemble'][best_model_name]
            self.best_model_name = best_model_name
            self.selected_features = selected_features
            self.is_trained = True
        
        training_results = {
            'models': model_results,
            'best_model': best_model_name if model_results else None,
            'selected_features': selected_features,
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'feature_count': len(selected_features)
        }
        
        return training_results
    
    def predict_signal_probability(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Predict signal probability using the trained model
        Returns probability and confidence metrics
        """
        if not self.is_trained:
            return {
                'signal_probability': 0.0,
                'prediction_confidence': 0.0,
                'model_used': 'None',
                'features_used': []
            }
        
        try:
            # Prepare features
            features = self.prepare_features(df)
            if features.empty:
                return {
                    'signal_probability': 0.0,
                    'prediction_confidence': 0.0,
                    'model_used': 'None',
                    'features_used': []
                }
            
            # Create labels for feature selection
            labels = self.create_labels(features)
            
            # Select features
            selected_features = self.select_features(features, labels)
            if not selected_features:
                return {
                    'signal_probability': 0.0,
                    'prediction_confidence': 0.0,
                    'model_used': 'None',
                    'features_used': []
                }
            
            # Clean NaN values
            feature_data = features[selected_features].fillna(method='ffill').fillna(method='bfill')
            if feature_data.isna().any().any():
                return {
                    'signal_probability': 0.0,
                    'prediction_confidence': 0.0,
                    'model_used': 'None',
                    'features_used': []
                }
            
            # Get predictions from all models
            predictions = {}
            confidences = {}
            
            # Ensemble predictions
            for name, model in self.models['ensemble'].items():
                try:
                    # Get model's prediction
                    pred = model.predict_proba(feature_data)[0]
                    predictions[name] = pred[1]  # Probability of positive class
                    
                    # Calculate confidence based on model's performance
                    if name in self.feature_importance:
                        metrics = self.feature_importance[name]
                        # Combine accuracy and precision for confidence
                        confidences[name] = (metrics['accuracy'] + metrics['precision']) / 2
                    else:
                        # Default confidence if metrics not available
                        confidences[name] = 0.5
                except Exception as e:
                    print(f"Error in model {name} prediction: {e}")
                    continue
            
            # Breakout predictions
            breakout_probs = []
            for name, model in self.models['breakout'].items():
                try:
                    pred = model.predict_proba(feature_data)[0]
                    breakout_probs.append(pred[1])
                except Exception as e:
                    print(f"Error in breakout model {name} prediction: {e}")
                    continue
            
            if not predictions:
                return {
                    'signal_probability': 0.0,
                    'prediction_confidence': 0.0,
                    'model_used': 'None',
                    'features_used': []
                }
            
            # Use the best performing model's prediction
            best_model = max(confidences.items(), key=lambda x: x[1])[0]
            signal_prob = predictions[best_model]
            confidence = confidences[best_model]
            
            # Get feature importance for the best model
            feature_importance = {}
            if best_model in self.feature_importance:
                feature_importance = self.feature_importance[best_model]
            elif hasattr(self.models['ensemble'][best_model], 'feature_importances_'):
                feature_importance = dict(zip(selected_features, 
                                           self.models['ensemble'][best_model].feature_importances_))
            
            top_features = sorted(
                zip(selected_features, [feature_importance.get(f, 0) for f in selected_features]),
                key=lambda x: x[1],
                reverse=True
            )[:5]  # Top 5 features
            
            # Combine predictions
            final_prob = signal_prob
            if breakout_probs:
                final_prob = (final_prob + np.mean(breakout_probs)) / 2
            
            return {
                'signal_probability': round(final_prob * 100, 1),
                'prediction_confidence': round(confidence * 100, 1),
                'model_used': best_model,
                'features_used': [f[0] for f in top_features],
                'breakout_probability': round(np.mean(breakout_probs) * 100, 1) if breakout_probs else 0.0
            }
            
        except Exception as e:
            print(f"Error in predict_signal_probability: {e}")
            return {
                'signal_probability': 0.0,
                'prediction_confidence': 0.0,
                'model_used': 'None',
                'features_used': []
            }
    
    def screen_with_ml(self, symbols: List[str], 
                      min_probability: float = 0.6) -> List[Dict]:
        """
        Screen symbols using ML model predictions
        
        Args:
            symbols: List of symbols to screen
            min_probability: Minimum probability threshold
            
        Returns:
            List of symbols with high prediction probability
        """
        if not self.is_trained:
            return []
        
        ml_candidates = []
        
        for symbol in symbols:
            try:
                df = get_stock_data(symbol, period="1y", interval="1d")
                if df.empty:
                    continue
                
                prediction = self.predict_signal_probability(df)
                
                if prediction['signal_probability'] >= min_probability * 100:
                    ml_candidates.append({
                        'symbol': symbol,
                        'probability': prediction['signal_probability'],
                        'confidence': prediction['prediction_confidence'],
                        'breakout_probability': prediction.get('breakout_probability', 0.0),
                        'trend_probability': prediction.get('trend_probability', 0.0),
                        'model_used': prediction['model_used'],
                        'top_features': prediction['features_used']
                    })
            except:
                continue
        
        # Sort by probability
        ml_candidates.sort(key=lambda x: x['probability'], reverse=True)
        
        return ml_candidates
    
    def get_feature_importance_ranking(self) -> Dict[str, float]:
        """
        Get feature importance ranking from trained models
        
        Returns:
            Dictionary with feature importance scores
        """
        if not self.is_trained or self.best_model_name not in self.models['ensemble']:
            return {}
        
        model = self.models['ensemble'][self.best_model_name]
        
        if hasattr(model, 'feature_importances_'):
            importance_dict = dict(zip(self.selected_features, model.feature_importances_))
            # Sort by importance
            sorted_importance = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
            return sorted_importance
        
        return {}
    
    def predict_price_direction(self, symbol: str, 
                              horizon_days: int = 5) -> Dict[str, Any]:
        """
        Predict price direction for a specific horizon
        
        Args:
            symbol: Stock symbol
            horizon_days: Prediction horizon in days
            
        Returns:
            Dictionary with direction prediction and confidence
        """
        prediction = self.predict_signal_probability(symbol)
        
        if 'error' in prediction:
            return prediction
        
        # Enhance with directional information
        direction = 'UP' if prediction['prediction'] == 1 else 'DOWN'
        strength = 'STRONG' if prediction['confidence'] > 0.7 else 'MODERATE' if prediction['confidence'] > 0.4 else 'WEAK'
        
        return {
            'symbol': symbol,
            'direction': direction,
            'strength': strength,
            'probability': prediction['probability'],
            'confidence': prediction['confidence'],
            'horizon_days': horizon_days,
            'model_used': prediction['model_used']
        }
    
    def backtest_model(self, symbol: str, 
                      start_date: str = None, 
                      end_date: str = None) -> Dict[str, float]:
        """
        Backtest the ML model on historical data
        
        Args:
            symbol: Stock symbol to backtest
            start_date: Start date for backtest
            end_date: End date for backtest
            
        Returns:
            Dictionary with backtest results
        """
        if not self.is_trained:
            return {'error': 'Model not trained'}
        
        try:
            # Get historical data for 10 years
            df = get_stock_data(symbol, period="10y", interval="1d")
            if df.empty or len(df) < 100:
                return {'error': 'Insufficient data for backtest'}
            
            # Prepare features and labels
            features_df = self.prepare_features(df)
            labels = self.create_labels(features_df)
            
            # Get valid data
            feature_data = features_df[self.selected_features].iloc[:-5]
            label_data = labels.iloc[:-5]
            
            valid_mask = ~(feature_data.isna().any(axis=1) | label_data.isna())
            X_test = feature_data[valid_mask]
            y_test = label_data[valid_mask]
            
            if len(X_test) < 50:
                return {'error': 'Insufficient valid data for backtest'}
            
            # Scale features
            if self.scaler:
                X_test_scaled = self.scaler.transform(X_test)
            else:
                X_test_scaled = X_test
            
            # Make predictions
            y_pred = self.best_model.predict(X_test_scaled)
            y_pred_proba = self.best_model.predict_proba(X_test_scaled)[:, 1]
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            
            # Calculate returns if trading based on predictions
            price_data = df['Close'].iloc[:-5][valid_mask]
            returns = price_data.pct_change(5).shift(-5)  # 5-day forward returns
            
            # Strategy returns (buy when prediction = 1)
            strategy_returns = returns * y_pred
            total_return = (1 + strategy_returns).prod() - 1
            
            # Calculate drawdown
            cumulative_returns = (1 + strategy_returns).cumprod()
            rolling_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns / rolling_max - 1).min()
            
            # Calculate annualized metrics
            years = len(price_data) / 252  # Assuming 252 trading days per year
            annualized_return = (1 + total_return) ** (1/years) - 1
            
            return {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'total_return': total_return,
                'annualized_return': annualized_return,
                'max_drawdown': drawdown,
                'win_rate': (strategy_returns > 0).mean(),
                'avg_win': strategy_returns[strategy_returns > 0].mean(),
                'avg_loss': strategy_returns[strategy_returns < 0].mean(),
                'sharpe_ratio': strategy_returns.mean() / strategy_returns.std() * np.sqrt(252),
                'positive_predictions': sum(y_pred),
                'avg_prediction_confidence': np.mean(y_pred_proba),
                'backtest_period_years': years
            }
            
        except Exception as e:
            return {'error': f'Backtest failed: {str(e)}'}
    
    def save_model_summary(self) -> Dict[str, Any]:
        """
        Save model summary for persistence
        
        Returns:
            Dictionary with model summary information
        """
        if not self.is_trained:
            return {'error': 'No trained model to save'}
        
        summary = {
            'is_trained': self.is_trained,
            'best_model_name': self.best_model_name,
            'selected_features': self.selected_features,
            'feature_importance': self.get_feature_importance_ranking(),
            'model_available': HAS_SKLEARN,
            'xgboost_available': HAS_XGBOOST,
            'lightgbm_available': HAS_LIGHTGBM,
            'tensorflow_available': HAS_TENSORFLOW,
            'training_timestamp': datetime.now().isoformat()
        }
        
        return summary 