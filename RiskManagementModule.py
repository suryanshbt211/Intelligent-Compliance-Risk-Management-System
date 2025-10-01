# icrms/modules/risk_management/risk_analyzer.py
"""
Financial Risk Management Module
Uses Prophet for forecasting and scikit-learn for risk modeling
"""

import numpy as np
import pandas as pd
from prophet import Prophet
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import norm
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)


class RiskAnalyzer:
    """Analyze financial and operational risk"""
    
    def __init__(self):
        self.var_confidence = 0.95
        self.forecaster = None
        
    def calculate_var(self, returns: np.ndarray, confidence: float = 0.95) -> float:
        """Calculate Value at Risk (VaR)"""
        try:
            var = np.percentile(returns, (1 - confidence) * 100)
            return float(var)
        except Exception as e:
            logger.error(f"VaR calculation error: {e}")
            return 0.0
    
    def calculate_cvar(self, returns: np.ndarray, confidence: float = 0.95) -> float:
        """Calculate Conditional Value at Risk (CVaR)"""
        try:
            var = self.calculate_var(returns, confidence)
            cvar = returns[returns <= var].mean()
            return float(cvar)
        except Exception as e:
            logger.error(f"CVaR calculation error: {e}")
            return 0.0
    
    def stress_test(self, portfolio_data: pd.DataFrame, 
                   scenarios: List[Dict]) -> Dict[str, any]:
        """Perform stress testing on portfolio"""
        results = {}
        
        for scenario in scenarios:
            name = scenario.get("name", "Unknown")
            shock = scenario.get("shock", 0.0)
            
            # Apply shock to returns
            stressed_returns = portfolio_data['returns'] * (1 + shock)
            
            results[name] = {
                "scenario": name,
                "shock": shock,
                "original_mean": float(portfolio_data['returns'].mean()),
                "stressed_mean": float(stressed_returns.mean()),
                "original_volatility": float(portfolio_data['returns'].std()),
                "stressed_volatility": float(stressed_returns.std()),
                "var_original": self.calculate_var(portfolio_data['returns'].values),
                "var_stressed": self.calculate_var(stressed_returns.values)
            }
        
        return results
    
    def forecast_risk(self, historical_data: pd.DataFrame, 
                     periods: int = 30) -> Dict[str, any]:
        """Forecast future risk using Prophet"""
        try:
            # Prepare data for Prophet
            df = historical_data.copy()
            df.columns = ['ds', 'y']
            
            # Initialize and fit Prophet
            model = Prophet(
                daily_seasonality=True,
                yearly_seasonality=False,
                changepoint_prior_scale=0.05
            )
            model.fit(df)
            
            # Make forecast
            future = model.make_future_dataframe(periods=periods)
            forecast = model.predict(future)
            
            return {
                "forecast": forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods).to_dict('records'),
                "trend": "increasing" if forecast['trend'].iloc[-1] > forecast['trend'].iloc[0] else "decreasing",
                "uncertainty": float(forecast['yhat_upper'].iloc[-1] - forecast['yhat_lower'].iloc[-1])
            }
        except Exception as e:
            logger.error(f"Risk forecasting error: {e}")
            return {"error": str(e)}
    
    def calculate_portfolio_metrics(self, portfolio_data: pd.DataFrame) -> Dict:
        """Calculate comprehensive portfolio risk metrics"""
        returns = portfolio_data['returns'].values
        
        return {
            "mean_return": float(np.mean(returns)),
            "volatility": float(np.std(returns)),
            "sharpe_ratio": float(np.mean(returns) / np.std(returns)) if np.std(returns) > 0 else 0.0,
            "var_95": self.calculate_var(returns, 0.95),
            "var_99": self.calculate_var(returns, 0.99),
            "cvar_95": self.calculate_cvar(returns, 0.95),
            "max_drawdown": float(self._calculate_max_drawdown(returns)),
            "skewness": float(pd.Series(returns).skew()),
            "kurtosis": float(pd.Series(returns).kurtosis())
        }
    
    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """Calculate maximum drawdown"""
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        return float(np.min(drawdown))


class CreditRiskModel:
    """Credit risk assessment model"""
    
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.is_trained = False
        
    def train(self, X: pd.DataFrame, y: np.ndarray):
        """Train credit risk model"""
        self.model.fit(X, y)
        self.is_trained = True
        logger.info("Credit risk model trained successfully")
    
    def predict_default_probability(self, customer_data: pd.DataFrame) -> np.ndarray:
        """Predict probability of default"""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        probabilities = self.model.predict_proba(customer_data)[:, 1]
        return probabilities
    
    def assess_credit_risk(self, customer_data: Dict) -> Dict:
        """Assess credit risk for a customer"""
        # Extract features
        features = pd.DataFrame([{
            'income': customer_data.get('income', 0),
            'debt_ratio': customer_data.get('debt_ratio', 0),
            'credit_score': customer_data.get('credit_score', 0),
            'employment_length': customer_data.get('employment_length', 0),
            'loan_amount': customer_data.get('loan_amount', 0)
        }])
        
        if self.is_trained:
            default_prob = self.predict_default_probability(features)[0]
        else:
            # Simple heuristic if model not trained
            default_prob = self._heuristic_risk_score(customer_data)
        
        risk_level = "LOW" if default_prob < 0.3 else "MEDIUM" if default_prob < 0.7 else "HIGH"
        
        return {
            "customer_id": customer_data.get("customer_id"),
            "default_probability": float(default_prob),
            "risk_level": risk_level,
            "recommended_action": self._get_recommendation(risk_level)
        }
    
    def _heuristic_risk_score(self, customer_data: Dict) -> float:
        """Calculate risk score using heuristics"""
        score = 0.5  # baseline
        
        if customer_data.get('credit_score', 0) < 600:
            score += 0.3
        elif customer_data.get('credit_score', 0) > 750:
            score -= 0.2
        
        if customer_data.get('debt_ratio', 0) > 0.5:
            score += 0.2
        
        return min(max(score, 0.0), 1.0)
    
    def _get_recommendation(self, risk_level: str) -> str:
        """Get recommendation based on risk level"""
        recommendations = {
            "LOW": "Approve with standard terms",
            "MEDIUM": "Approve with enhanced monitoring",
            "HIGH": "Reject or require additional collateral"
        }
        return recommendations.get(risk_level, "Manual review required")


# icrms/modules/risk_management/__init__.py
from .risk_analyzer import RiskAnalyzer, CreditRiskModel

__all__ = ['RiskAnalyzer', 'CreditRiskModel']
