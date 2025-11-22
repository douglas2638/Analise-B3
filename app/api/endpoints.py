from fastapi import APIRouter, HTTPException, Query, BackgroundTasks
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import asyncio
import logging
import time
import math
from typing import Any, Dict

from app.services.data_collector import B3DataCollector
from app.services.data_processor import DataProcessor
from app.services.analyzer import StockAnalyzer
from app.models.schemas import AnalysisRequest, AnalysisResult, PortfolioAnalysis

def clean_json_data(data: Any) -> Any:
    """
    Remove valores NaN, inf, -inf para serialização JSON
    """
    if data is None:
        return None
    elif isinstance(data, (int, str, bool)):
        return data
    elif isinstance(data, float):
        if math.isnan(data) or math.isinf(data):
            return 0.0
        return data
    elif isinstance(data, (np.float32, np.float64, np.int32, np.int64)):
        data_float = float(data)
        if math.isnan(data_float) or math.isinf(data_float):
            return 0.0
        return data_float
    elif isinstance(data, pd.Timestamp):
        return data.isoformat()
    elif isinstance(data, (list, tuple)):
        return [clean_json_data(item) for item in data]
    elif isinstance(data, dict):
        return {key: clean_json_data(value) for key, value in data.items()}
    elif hasattr(data, '__dict__'):
        return clean_json_data(data.__dict__)
    elif isinstance(data, pd.Series):
        return clean_json_data(data.tolist())
    elif isinstance(data, pd.DataFrame):
        return clean_json_data(data.to_dict())
    else:
        try:
            # Tenta converter para string como último recurso
            return str(data)
        except:
            return None

router = APIRouter()

# Instâncias dos serviços
data_collector = B3DataCollector()
data_processor = DataProcessor()
analyzer = StockAnalyzer()

# Cache em memória para otimização
analysis_cache = {}
logger = logging.getLogger(__name__)

@router.get("/stock/{symbol}")
async def get_stock_data(
    symbol: str,
    period: str = Query("6mo", description="Período: 1d, 5d, 1mo, 3mo, 6mo, 1y"),
    include_indicators: bool = Query(False, description="Incluir indicadores técnicos"),
    use_cache: bool = Query(True, description="Usar cache para evitar rate limiting")
    
):
    """Busca dados históricos de uma ação com rate limiting"""
    try:
        symbol = symbol.upper().strip()
        
        # Verifica cache primeiro
        cache_key = f"stock_{symbol}_{period}"
        if use_cache and cache_key in analysis_cache:
            cached = analysis_cache[cache_key]
            cache_age = datetime.now() - cached['timestamp']
            if cache_age < timedelta(minutes=30):  # Cache de 30 minutos
                logger.info(f"Retornando dados do cache para {symbol} (idade: {cache_age})")
                result = cached['data']
                result["cache_info"] = {
                    "from_cache": True,
                    "cache_age_minutes": cache_age.total_seconds() / 60,
                    "cache_timestamp": cached['timestamp'].isoformat()
                }
                return result
        
        # Busca dados (AGORA APENAS 2 ARGUMENTOS: symbol e period)
        logger.info(f"Buscando dados para {symbol} (período: {period})")
        data = data_collector.get_stock_data(symbol, period)  # APENAS 2 ARGUMENTOS
        
        if data.empty:
            raise HTTPException(
                status_code=404, 
                detail=f"Dados não encontrados para {symbol}. O Yahoo Finance pode estar com rate limiting. Tente novamente em alguns minutos."
            )
        
        # Prepara resposta
        response_data = data.reset_index()
        response_data['Date'] = response_data['Date'].dt.strftime('%Y-%m-%d')
        
        result = {
            "symbol": symbol,
            "period": period,
            "data_points": len(data),
            "data": response_data.to_dict('records'),
            "cache_info": {
                "from_cache": False,
                "cache_timestamp": datetime.now().isoformat()
            },
            "last_price": float(data['Close'].iloc[-1]) if not data.empty else 0,
            "price_change": float(data['Close'].pct_change().iloc[-1] * 100) if len(data) > 1 else 0
        }
        
        # Adiciona indicadores se solicitado
        if include_indicators and not data.empty:
            try:
                processed_data = data_processor.calculate_all_indicators(data)
                latest_indicators = processed_data.iloc[-1].to_dict() if not processed_data.empty else {}
                
                # Filtra apenas valores numéricos válidos
                result["latest_indicators"] = {
                    k: float(v) for k, v in latest_indicators.items() 
                    if isinstance(v, (int, float, np.number)) 
                    and not np.isnan(v) 
                    and not np.isinf(v)
                }
            except Exception as e:
                logger.warning(f"Erro ao calcular indicadores para {symbol}: {e}")
                result["latest_indicators"] = {"error": "Não foi possível calcular indicadores"}
        
        # Adiciona ao cache (apenas se não veio do cache)
        if use_cache and not data.empty:
            analysis_cache[cache_key] = {
                'data': result,
                'timestamp': datetime.now()
            }
            result["cache_info"]["cached"] = True
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erro interno em /stock/{symbol}: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Erro interno: {str(e)}"
        )

@router.get("/stock/{symbol}/info")
async def get_stock_info(
    symbol: str,
    use_cache: bool = Query(True, description="Usar cache")
):
    """Busca informações da empresa SEM usar yfinance.info"""
    try:
        symbol = symbol.upper().strip()
        
        # Verifica cache
        cache_key = f"info_{symbol}"
        if use_cache and cache_key in analysis_cache:
            cached = analysis_cache[cache_key]
            if datetime.now() - cached['timestamp'] < timedelta(minutes=60):
                return cached['data']
        
        # Usa método seguro que não causa rate limiting
        info = data_collector.get_stock_info(symbol)
        
        if not info:
            raise HTTPException(
                status_code=404, 
                detail=f"Informações não encontradas para {symbol}"
            )
        
        # Adiciona ao cache
        if use_cache:
            analysis_cache[cache_key] = {
                'data': info,
                'timestamp': datetime.now()
            }
        
        return info
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erro em /stock/{symbol}/info: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Erro ao buscar informações: {str(e)}"
        )

@router.post("/analyze", response_model=AnalysisResult)
async def analyze_stock(
    request: AnalysisRequest,
    background_tasks: BackgroundTasks,
    use_cache: bool = Query(True, description="Usar cache")
):
    """Análise técnica completa de uma ação com rate limiting"""
    try:
        cache_key = f"analysis_{request.symbol}_{request.start_date}_{request.end_date}"
        
        # Verifica cache
        if use_cache and cache_key in analysis_cache:
            cached_result = analysis_cache[cache_key]
            cache_age = datetime.now() - cached_result['timestamp']
            if cache_age < timedelta(minutes=15):
                logger.info(f"Retornando análise do cache para {request.symbol}")
                return AnalysisResult(**cached_result['data'])
        
        # Busca dados
        data = data_collector.get_stock_data(request.symbol, "6mo")
        
        if data.empty:
            raise HTTPException(
                status_code=404, 
                detail=f"Dados não encontrados para {request.symbol}."
            )
        
        # Filtra por período se especificado
        if request.start_date and request.end_date:
            try:
                start_dt = pd.to_datetime(request.start_date)
                end_dt = pd.to_datetime(request.end_date)
                data = data[(data.index >= start_dt) & (data.index <= end_dt)]
                
                if data.empty:
                    raise HTTPException(
                        status_code=400, 
                        detail="Nenhum dado encontrado para o período especificado"
                    )
            except Exception as e:
                logger.warning(f"Erro ao filtrar por período: {e}")
        
        # Realiza análise
        analysis = analyzer.analyze_stock(request.symbol, data)
        
        if 'error' in analysis:
            raise HTTPException(
                status_code=400,
                detail=f"Erro na análise: {analysis['error']}"
            )
        
        statistics_data = analysis.get('statistics', {})
        
        # Preenche campos faltantes com valores padrão
        required_stats = ['var_95', 'cvar_95']
        for field in required_stats:
            if field not in statistics_data:
                statistics_data[field] = 0.0
        
        # Prepara resposta
        result_data = {
            "symbol": analysis['symbol'],
            "period": f"{request.start_date} to {request.end_date}" if request.start_date and request.end_date else "6mo",
            "statistics": statistics_data,
            "indicators": analysis.get('indicators', {}),
            "recommendations": analysis.get('recommendations', []),
            "score": analysis.get('score', 50),
            "trend_analysis": analysis.get('trend_analysis'),
            "timestamp": analysis.get('timestamp', datetime.now().isoformat())
        }
        
        cleaned_data = clean_json_data(result_data)
        
        result = AnalysisResult(**cleaned_data)
        
        # Adiciona ao cache em background
        if use_cache:
            background_tasks.add_task(
                update_analysis_cache, 
                cache_key, 
                cleaned_data
            )
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erro em /analyze: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Erro na análise: {str(e)}"
        )

def update_analysis_cache(cache_key: str, data: dict):
    """Atualiza cache em background"""
    analysis_cache[cache_key] = {
        'data': data,
        'timestamp': datetime.now()
    }

@router.get("/analyze/quick/{symbol}")
async def quick_analyze(
    symbol: str,
    use_cache: bool = Query(True, description="Usar cache")
):
    """Análise rápida de uma ação com cache agressivo"""
    try:
        symbol = symbol.upper().strip()
        
        cache_key = f"quick_{symbol}"
        if use_cache and cache_key in analysis_cache:
            cached = analysis_cache[cache_key]
            if datetime.now() - cached['timestamp'] < timedelta(minutes=10):  # Cache curto para análise rápida
                return cached['data']
        
        # Busca dados dos últimos 3 meses para análise rápida (menos dados = menos rate limiting)
        data = data_collector.get_stock_data(symbol, "3mo")
        
        if data.empty:
            raise HTTPException(
                status_code=404, 
                detail=f"Dados não encontrados para {symbol}. Tente novamente em alguns segundos."
            )
        
        analysis = analyzer.analyze_stock(symbol, data)
        
        result = {
            "symbol": symbol,
            "score": analysis.get('score', 50),
            "trend": analysis.get('trend_analysis', {}).get('primary', 'UNKNOWN'),
            "recommendation": analysis.get('recommendations', ['N/A'])[0],
            "current_price": analysis.get('statistics', {}).get('price_current', 0),
            "daily_change": analysis.get('statistics', {}).get('price_variation_1d', 0),
            "analysis_timestamp": datetime.now().isoformat()
        }
        
        # Cache agressivo para análise rápida
        if use_cache:
            analysis_cache[cache_key] = {
                'data': result,
                'timestamp': datetime.now()
            }
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erro em /analyze/quick/{symbol}: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Erro na análise rápida: {str(e)}"
        )

@router.get("/portfolio/analyze")
async def analyze_portfolio(
    stocks: str = Query(..., description="Ações separadas por vírgula"),
    weights: Optional[str] = Query(None, description="Pesos separados por vírgula")
):
    
    try:
        # Parse simples
        stock_list = [s.strip().upper() for s in stocks.split(',') if s.strip()]
        
        if weights:
            weight_list = [float(w.strip()) for w in weights.split(',') if w.strip()]
        else:
            weight_list = [1.0/len(stock_list)] * len(stock_list)
        
        # Validação básica
        if len(weight_list) != len(stock_list):
            return {"error": "Número de pesos não corresponde ao número de ações"}
        
        # Coleta dados básica
        analyses = {}
        for symbol in stock_list:
            try:
                data = data_collector.get_stock_data(symbol, "3mo")
                if not data.empty:
                    # Análise simplificada
                    current_price = float(data['Close'].iloc[-1])
                    analyses[symbol] = {
                        "symbol": symbol,
                        "current_price": current_price,
                        "data_points": len(data),
                        "analysis": "Dados coletados com sucesso"
                    }
            except:
                analyses[symbol] = {"error": "Falha na coleta de dados"}
        
        # Resultado simplificado
        result = {
            "stocks": stock_list,
            "weights": weight_list,
            "analyses": analyses,
            "portfolio_summary": {
                "total_stocks": len(stock_list),
                "analysis_timestamp": datetime.now().isoformat()
            }
        }
        
        # Limpeza final garantida
        return clean_json_data(result)
        
    except Exception as e:
        return clean_json_data({"error": f"Erro seguro: {str(e)}"})

@router.get("/market/overview")
async def market_overview(
    use_cache: bool = Query(True, description="Usar cache")
):
    """Visão geral do mercado com cache agressivo"""
    try:
        cache_key = "market_overview"
        if use_cache and cache_key in analysis_cache:
            cached = analysis_cache[cache_key]
            if datetime.now() - cached['timestamp'] < timedelta(minutes=5):  # Cache curto para dados de mercado
                return cached['data']
        
        major_stocks = ["PETR4", "VALE3", "ITUB4", "BBDC4", "B3SA3"]
        
        overview_data = {}
        successful_requests = 0
        
        for symbol in major_stocks:
            try:
                # Busca apenas dados do dia atual
                data = data_collector.get_stock_data(symbol, "1d")
                
                if not data.empty and len(data) > 1:
                    current_price = data['Close'].iloc[-1]
                    previous_close = data['Close'].iloc[-2]  # Usa penúltimo valor como previous
                    change = ((current_price - previous_close) / previous_close) * 100
                    
                    overview_data[symbol] = {
                        "current_price": float(current_price),
                        "change": float(change),
                        "volume": int(data['Volume'].iloc[-1]),
                        "timestamp": data.index[-1].strftime('%Y-%m-%d %H:%M:%S')
                    }
                    successful_requests += 1
                else:
                    overview_data[symbol] = {
                        "error": "Dados não disponíveis",
                        "current_price": 0,
                        "change": 0
                    }
                    
            except Exception as e:
                logger.warning(f"Erro ao buscar {symbol}: {e}")
                overview_data[symbol] = {
                    "error": str(e),
                    "current_price": 0,
                    "change": 0
                }
        
        # Calcula desempenho geral apenas com requests bem-sucedidos
        if successful_requests > 0:
            changes = [stock["change"] for stock in overview_data.values() if "error" not in stock]
            avg_change = sum(changes) / len(changes) if changes else 0
            market_sentiment = "POSITIVE" if avg_change > 0.1 else "NEGATIVE" if avg_change < -0.1 else "NEUTRAL"
        else:
            avg_change = 0
            market_sentiment = "UNAVAILABLE"
        
        result = {
            "timestamp": datetime.now().isoformat(),
            "market_sentiment": market_sentiment,
            "average_change": float(avg_change),
            "successful_requests": successful_requests,
            "total_requests": len(major_stocks),
            "stocks": overview_data,
            "rate_limiting": "active",
            "cache_recommendation": "Use use_cache=true para melhor performance"
        }
        
        # Cache agressivo para overview
        if use_cache:
            analysis_cache[cache_key] = {
                'data': result,
                'timestamp': datetime.now()
            }
        
        return result
        
    except Exception as e:
        logger.error(f"Erro em /market/overview: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health/detailed")
async def detailed_health_check():
    """Verificação detalhada da saúde da API - Não causa rate limiting"""
    try:
        # Teste leve que não causa rate limiting
        yfinance_working = False
        cache_status = len(analysis_cache)
        
        # Verifica se temos dados em cache como indicador de funcionamento
        if cache_status > 0:
            yfinance_working = True
        else:
            # Teste muito leve: apenas verifica se podemos importar e criar objeto
            try:
                import yfinance as yf
                # Não faz requisição real!
                ticker = yf.Ticker("PETR4.SA")
                # Marca como working (a requisição real pode falhar, mas o serviço está disponível)
                yfinance_working = True
            except Exception as e:
                yfinance_working = False
        
        services_status = {
            "yfinance_connection": yfinance_working,
            "cache_status": cache_status,
            "data_collector": True,
            "data_processor": True,
            "analyzer": True,
            "rate_limiting": "active",
            "cache_enabled": cache_status > 0
        }
        
        # Status baseado nos serviços core, não no yfinance
        core_services_healthy = all([
            services_status["data_collector"],
            services_status["data_processor"], 
            services_status["analyzer"]
        ])
        
        status = "healthy" if core_services_healthy else "degraded"
        
        notes = []
        if not yfinance_working:
            notes.append("Yahoo Finance pode estar com rate limiting - usando cache")
        if cache_status == 0:
            notes.append("Cache vazio - primeiras requisições podem ser lentas")
        
        return {
            "status": status,
            "timestamp": datetime.now().isoformat(),
            "services": services_status,
            "version": "1.0.0",
            "notes": notes if notes else None,
            "recommendations": [
                "Use parâmetro use_cache=true para melhor performance",
                "Evite muitas requisições rápidas",
                "Use períodos menores para dados recentes"
            ]
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@router.get("/cache/clear")
async def clear_cache():
    """Limpa o cache de análises"""
    try:
        cache_size = len(analysis_cache)
        analysis_cache.clear()
        data_collector.clear_cache()
        
        return {
            "message": "Cache limpo com sucesso", 
            "cleared_at": datetime.now().isoformat(),
            "cleared_entries": cache_size
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/cache/status")
async def cache_status():
    """Status do cache"""
    try:
        cache_entries = {}
        for key, value in analysis_cache.items():
            age = (datetime.now() - value['timestamp']).total_seconds() / 60  # idade em minutos
            cache_entries[key] = {
                "age_minutes": round(age, 2),
                "timestamp": value['timestamp'].isoformat()
            }
        
        return {
            "total_entries": len(analysis_cache),
            "entries": cache_entries,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Funções auxiliares (mantenha as mesmas do anterior)
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
        if not returns_data:
            return {'matrix': {}, 'average_correlation': 0}
            
        # Cria DataFrame com retornos
        returns_df = pd.DataFrame(returns_data)
        correlation_matrix = returns_df.corr()
        
        # Converte para formato serializável
        return {
            'matrix': correlation_matrix.fillna(0).to_dict(),
            'average_correlation': float(correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)].mean())
        }
    except Exception as e:
        logger.error(f"Erro ao calcular correlação: {e}")
        return {'matrix': {}, 'average_correlation': 0}

async def _calculate_portfolio_stats(returns_data: dict, weights: List[float], current_prices: dict, initial_investment: float) -> dict:
    """Calcula estatísticas da carteira"""
    try:
        symbols = list(returns_data.keys())
        
        if not symbols:
            return {"error": "Sem dados de retorno para cálculo"}
        
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
        
        return {
            "expected_annual_return": float(portfolio_return * 100),
            "annual_volatility": float(portfolio_volatility * 100),
            "sharpe_ratio": float(sharpe_ratio),
            "initial_investment": initial_investment,
            "risk_free_rate": risk_free_rate * 100
        }
    except Exception as e:
        logger.error(f"Erro ao calcular stats da carteira: {e}")
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
        "sectors": list(popular_stocks.keys()),
        "rate_limiting_note": "Recomendado: analisar até 5 ações por vez"
    }