from fastapi import APIRouter, HTTPException, Query, BackgroundTasks
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import asyncio

from app.services.data_collector import B3DataCollector
from app.services.data_processor import DataProcessor
from app.services.analyzer import StockAnalyzer
from app.models.schemas import AnalysisRequest, AnalysisResult, PortfolioAnalysis

router = APIRouter()

# Instâncias dos serviços
data_collector = B3DataCollector()
data_processor = DataProcessor()
analyzer = StockAnalyzer(data_processor)

# Cache em memória para otimização
analysis_cache = {}

@router.get("/stock/{symbol}")
async def get_stock_data(
    symbol: str,
    period: str = Query("6mo", description="Período: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max"),
    include_indicators: bool = Query(False, description="Incluir indicadores técnicos")
):
    """Busca dados históricos de uma ação"""
    try:
        # Valida símbolo
        symbol = symbol.upper().strip()
        
        # Busca dados
        data = data_collector.get_stock_data(symbol, period)
        if data.empty:
            raise HTTPException(status_code=404, detail=f"Dados não encontrados para {symbol}")
        
        # Prepara resposta
        response_data = data.reset_index()
        response_data['Date'] = response_data['Date'].dt.strftime('%Y-%m-%d')
        
        result = {
            "symbol": symbol,
            "period": period,
            "data_points": len(data),
            "data": response_data.to_dict('records')
        }
        
        # Adiciona indicadores se solicitado
        if include_indicators:
            processed_data = data_processor.calculate_all_indicators(data)
            latest_indicators = processed_data.iloc[-1].to_dict() if not processed_data.empty else {}
            result["latest_indicators"] = {
                k: float(v) for k, v in latest_indicators.items() 
                if isinstance(v, (int, float, np.number)) and not np.isnan(v)
            }
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")

@router.get("/stock/{symbol}/info")
async def get_stock_info(symbol: str):
    """Busca informações da empresa"""
    try:
        symbol = symbol.upper().strip()
        info = data_collector.get_stock_info(symbol)
        
        if not info:
            raise HTTPException(status_code=404, detail=f"Informações não encontradas para {symbol}")
        
        return info
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/analyze", response_model=AnalysisResult)
async def analyze_stock(request: AnalysisRequest):
    """Análise técnica completa de uma ação"""
    try:
        cache_key = f"{request.symbol}_{request.start_date}_{request.end_date}"
        
        # Verifica cache
        if cache_key in analysis_cache:
            cached_result = analysis_cache[cache_key]
            if datetime.now() - cached_result['timestamp'] < timedelta(minutes=10):
                return AnalysisResult(**cached_result['data'])
        
        # Busca dados
        data = data_collector.get_stock_data(request.symbol, "1y")
        if data.empty:
            raise HTTPException(status_code=404, detail=f"Dados não encontrados para {request.symbol}")
        
        # Filtra por período se especificado
        if request.start_date and request.end_date:
            try:
                start_dt = pd.to_datetime(request.start_date)
                end_dt = pd.to_datetime(request.end_date)
                data = data[(data.index >= start_dt) & (data.index <= end_dt)]
            except:
                pass  # Usa todos os dados se houver erro na conversão
        
        # Realiza análise
        analysis = analyzer.analyze_stock(request.symbol, data)
        
        # Prepara resposta
        result = AnalysisResult(
            symbol=analysis['symbol'],
            period=f"{request.start_date} to {request.end_date}" if request.start_date and request.end_date else "1y",
            statistics=analysis.get('statistics', {}),
            indicators=analysis.get('indicators', {}),
            recommendations=analysis.get('recommendations', [])
        )
        
        # Atualiza cache
        analysis_cache[cache_key] = {
            'data': result.dict(),
            'timestamp': datetime.now()
        }
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/analyze/quick/{symbol}")
async def quick_analyze(symbol: str):
    """Análise rápida de uma ação"""
    try:
        symbol = symbol.upper().strip()
        
        # Busca dados dos últimos 6 meses para análise rápida
        data = data_collector.get_stock_data(symbol, "6mo")
        if data.empty:
            raise HTTPException(status_code=404, detail=f"Dados não encontrados para {symbol}")
        
        analysis = analyzer.analyze_stock(symbol, data)
        
        return {
            "symbol": symbol,
            "score": analysis.get('score', 50),
            "trend": analysis.get('trend_analysis', {}).get('primary', 'UNKNOWN'),
            "recommendation": analysis.get('recommendations', ['N/A'])[0],
            "current_price": analysis.get('statistics', {}).get('price_current', 0),
            "daily_change": analysis.get('statistics', {}).get('price_variation_1d', 0)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/portfolio/analyze")
async def analyze_portfolio(
    stocks: List[str] = Query(..., description="Lista de ações (ex: PETR4,VALE3,ITUB4)"),
    weights: Optional[List[float]] = Query(None, description="Pesos da carteira"),
    initial_investment: float = Query(10000.0, description="Investimento inicial")
):
    """Análise de carteira de ações"""
    try:
        if not stocks:
            raise HTTPException(status_code=400, detail="Lista de ações não pode estar vazia")
        
        # Limita a 10 ações para performance
        stocks = stocks[:10]
        
        if weights and len(weights) != len(stocks):
            raise HTTPException(status_code=400, detail="Número de pesos não corresponde ao número de ações")
        
        if not weights:
            weights = [1/len(stocks)] * len(stocks)
        
        # Verifica se pesos somam 1
        if abs(sum(weights) - 1.0) > 0.01:
            weights = [w/sum(weights) for w in weights]
        
        # Coleta dados
        stocks_data = data_collector.get_multiple_stocks(stocks, "6mo")
        
        if not stocks_data:
            raise HTTPException(status_code=404, detail="Nenhum dado encontrado para as ações especificadas")
        
        # Análise individual e da carteira
        portfolio_analysis = await _analyze_portfolio_comprehensive(stocks_data, weights, initial_investment)
        
        return portfolio_analysis
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def _analyze_portfolio_comprehensive(stocks_data: dict, weights: List[float], initial_investment: float) -> dict:
    """Análise compreensiva de carteira"""
    analyses = {}
    returns_data = {}
    current_prices = {}
    
    # Análise individual
    for symbol, data in stocks_data.items():
        if not data.empty:
            analysis = analyzer.analyze_stock(symbol, data)
            analyses[symbol] = analysis
            
            # Calcula retornos
            data_with_returns = data_processor.calculate_returns(data)
            returns_data[symbol] = data_with_returns['daily_return'].dropna()
            current_prices[symbol] = data['Close'].iloc[-1]
    
    # Análise de correlação
    correlation_matrix = await _calculate_correlation_matrix(returns_data)
    
    # Cálculos de risco e retorno da carteira
    portfolio_stats = await _calculate_portfolio_stats(returns_data, weights, current_prices, initial_investment)
    
    return {
        "portfolio_statistics": portfolio_stats,
        "correlation_matrix": correlation_matrix,
        "individual_analyses": analyses,
        "weights": dict(zip(stocks_data.keys(), weights)),
        "diversification_score": await _calculate_diversification_score(correlation_matrix)
    }

async def _calculate_correlation_matrix(returns_data: dict) -> dict:
    """Calcula matriz de correlação"""
    try:
        # Cria DataFrame com retornos
        returns_df = pd.DataFrame(returns_data)
        correlation_matrix = returns_df.corr()
        
        # Converte para formato serializável
        return {
            'matrix': correlation_matrix.fillna(0).to_dict(),
            'average_correlation': float(correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)].mean())
        }
    except:
        return {'matrix': {}, 'average_correlation': 0}

async def _calculate_portfolio_stats(returns_data: dict, weights: List[float], current_prices: dict, initial_investment: float) -> dict:
    """Calcula estatísticas da carteira"""
    try:
        symbols = list(returns_data.keys())
        
        if not symbols:
            return {}
        
        # Retornos esperados
        expected_returns = []
        for symbol in symbols:
            if symbol in returns_data:
                ret = returns_data[symbol]
                expected_returns.append(ret.mean() * 252)
        
        portfolio_return = sum(w * ret for w, ret in zip(weights, expected_returns))
        
        # Risco da carteira
        returns_df = pd.DataFrame({symbol: returns_data[symbol] for symbol in symbols})
        cov_matrix = returns_df.cov() * 252
        portfolio_variance = np.dot(weights, np.dot(cov_matrix, weights))
        portfolio_volatility = np.sqrt(portfolio_variance)
        
        # Sharpe Ratio (assumindo risco livre 0.12 = 12% CDI)
        risk_free_rate = 0.12
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility if portfolio_volatility > 0 else 0
        
        # Valor atual da carteira
        portfolio_value = 0
        for symbol, weight in zip(symbols, weights):
            investment_amount = initial_investment * weight
            shares = investment_amount / list(stocks_data[symbol]['Close'])[0] if symbol in stocks_data and len(stocks_data[symbol]) > 0 else 0
            current_value = shares * current_prices.get(symbol, 0)
            portfolio_value += current_value
        
        total_return_pct = (portfolio_value / initial_investment - 1) * 100
        
        return {
            "expected_annual_return": float(portfolio_return * 100),
            "annual_volatility": float(portfolio_volatility * 100),
            "sharpe_ratio": float(sharpe_ratio),
            "initial_investment": initial_investment,
            "current_value": float(portfolio_value),
            "total_return": float(total_return_pct),
            "risk_free_rate": risk_free_rate * 100
        }
    except Exception as e:
        return {"error": str(e)}

async def _calculate_diversification_score(correlation_matrix: dict) -> float:
    """Calcula score de diversificação baseado na correlação"""
    try:
        avg_corr = correlation_matrix.get('average_correlation', 1)
        # Quanto menor a correlação média, melhor a diversificação
        diversification_score = max(0, 100 * (1 - abs(avg_corr)))
        return float(diversification_score)
    except:
        return 50.0

@router.get("/market/overview")
async def market_overview():
    """Visão geral do mercado com ações principais"""
    try:
        major_stocks = ["PETR4", "VALE3", "ITUB4", "BBDC4", "B3SA3", "WEGE3"]
        
        overview_data = {}
        for symbol in major_stocks:
            try:
                data = data_collector.get_stock_data(symbol, "1d")
                if not data.empty:
                    current_price = data['Close'].iloc[-1]
                    previous_close = data['Close'].iloc[0] if len(data) > 1 else current_price
                    change = ((current_price - previous_close) / previous_close) * 100
                    
                    overview_data[symbol] = {
                        "current_price": float(current_price),
                        "change": float(change),
                        "volume": int(data['Volume'].iloc[-1])
                    }
            except:
                continue
        
        # Calcula desempenho geral
        if overview_data:
            avg_change = sum(stock["change"] for stock in overview_data.values()) / len(overview_data)
            market_sentiment = "POSITIVE" if avg_change > 0 else "NEGATIVE" if avg_change < 0 else "NEUTRAL"
        else:
            avg_change = 0
            market_sentiment = "UNKNOWN"
        
        return {
            "timestamp": datetime.now().isoformat(),
            "market_sentiment": market_sentiment,
            "average_change": float(avg_change),
            "stocks": overview_data,
            "total_stocks_tracked": len(overview_data)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/stocks/list")
async def get_available_stocks():
    """Lista de ações disponíveis para análise com setores"""
    popular_stocks = {
        "Petróleo & Gás": ["PETR4", "PETR3", "PRIO3"],
        "Mineração": ["VALE3", "CSNA3"],
        "Bancos": ["ITUB4", "BBDC4", "BBAS3", "SANB11", "B3SA3"],
        "Varejo": ["MGLU3", "VIIA3", "AMER3"],
        "Energia": ["EQTL3", "ENGI11", "TAEE11"],
        "Industrial": ["WEGE3", "EMBR3", "RENT3"],
        "Saúde": ["HAPV3", "RADL3", "GNDI3"],
        "Consumo": ["ABEV3", "ASAI3", "LREN3"]
    }
    
    all_stocks = []
    for sector, stocks in popular_stocks.items():
        all_stocks.extend(stocks)
    
    return {
        "available_stocks": popular_stocks,
        "all_stocks": all_stocks,
        "total_count": len(all_stocks),
        "sectors": list(popular_stocks.keys())
    }

@router.get("/cache/clear")
async def clear_cache():
    """Limpa o cache de análises"""
    try:
        analysis_cache.clear()
        data_collector.clear_cache()
        return {"message": "Cache limpo com sucesso", "cleared_at": datetime.now().isoformat()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health/detailed")
async def detailed_health_check():
    """Verificação detalhada da saúde da API"""
    try:
        # Testa conexão com serviços externos
        test_symbol = "PETR4"
        test_data = data_collector.get_stock_data(test_symbol, "1d")
        
        services_status = {
            "yfinance_connection": not test_data.empty,
            "cache_status": len(analysis_cache),
            "data_collector": True,
            "data_processor": True,
            "analyzer": True
        }
        
        all_healthy = all(services_status.values())
        
        return {
            "status": "healthy" if all_healthy else "degraded",
            "timestamp": datetime.now().isoformat(),
            "services": services_status,
            "version": "1.0.0"
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }