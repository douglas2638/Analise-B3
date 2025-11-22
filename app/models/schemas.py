from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from enum import Enum

class TrendType(str, Enum):
    BULL_MARKET = "BULL_MARKET"
    BEAR_MARKET = "BEAR_MARKET"
    UPTREND = "UPTREND"
    DOWNTREND = "DOWNTREND"
    SIDEWAYS = "SIDEWAYS"
    CONSOLIDATION = "CONSOLIDATION"
    TRANSITION = "TRANSITION"

class AnalysisRequest(BaseModel):
    symbol: str = Field(..., description="Símbolo da ação (ex: PETR4)")
    start_date: Optional[str] = Field(None, description="Data de início (YYYY-MM-DD)")
    end_date: Optional[str] = Field(None, description="Data de fim (YYYY-MM-DD)")
    indicators: List[str] = Field(default=[], description="Indicadores técnicos específicos")

class StockData(BaseModel):
    symbol: str
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: int
    adj_close: Optional[float] = None

class TechnicalIndicators(BaseModel):
    moving_averages: Dict[str, float] = Field(default_factory=dict)
    rsi: Dict[str, float] = Field(default_factory=dict)
    bollinger_bands: Dict[str, Any] = Field(default_factory=dict)
    macd: Dict[str, Any] = Field(default_factory=dict)
    support_resistance: Dict[str, float] = Field(default_factory=dict)

class TrendAnalysis(BaseModel):
    primary_trend: str
    secondary_trend: str
    short_term_trend: str
    trend_strength: float
    key_levels: Dict[str, float]

# CORREÇÃO: Statistics com campos opcionais
class Statistics(BaseModel):
    price_current: float
    price_variation_1d: float
    total_return: float
    annual_return: float
    annual_volatility: float
    sharpe_ratio: float
    max_drawdown: float
    var_95: Optional[float] = Field(None, description="Value at Risk 95%")  # ← OPCIONAL
    cvar_95: Optional[float] = Field(None, description="Conditional Value at Risk 95%")  # ← OPCIONAL
    volume: Dict[str, Any]

class AnalysisResult(BaseModel):
    symbol: str
    period: str
    statistics: Statistics
    indicators: TechnicalIndicators
    trend_analysis: Optional[TrendAnalysis] = None
    recommendations: List[str]
    score: Optional[int] = None
    timestamp: Optional[str] = None

class PortfolioAnalysis(BaseModel):
    portfolio_statistics: Dict[str, Any]
    correlation_matrix: Dict[str, Any]
    individual_analyses: Dict[str, Any]
    weights: Dict[str, float]
    diversification_score: float

class MarketOverview(BaseModel):
    timestamp: str
    market_sentiment: str
    average_change: float
    stocks: Dict[str, Any]
    total_stocks_tracked: int

class HealthCheck(BaseModel):
    status: str
    timestamp: str
    services: Dict[str, bool]
    version: str