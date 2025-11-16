import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import requests
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class B3DataCollector:
    def __init__(self):
        self.base_url = "https://b3.com.br"
        self.cache = {}  # Cache simples para evitar múltiplas requisições
    
    def get_stock_data(self, symbol: str, period: str = "1y") -> pd.DataFrame:
        """
        Busca dados históricos de ações da B3
        """
        cache_key = f"{symbol}_{period}"
        
        if cache_key in self.cache:
            return self.cache[cache_key]
            
        try:
            # Adiciona .SA para ações brasileiras no yfinance
            ticker_symbol = f"{symbol}.SA"
            ticker = yf.Ticker(ticker_symbol)
            data = ticker.history(period=period)
            
            if data.empty:
                logger.warning(f"Dados vazios para {symbol}")
                return pd.DataFrame()
            
            # Limpa dados NaN
            data = data.dropna()
            
            # Adiciona símbolo como coluna
            data['Symbol'] = symbol
            
            # Cache por 5 minutos
            self.cache[cache_key] = data
            logger.info(f"Dados coletados para {symbol}, período: {period}")
            
            return data
            
        except Exception as e:
            logger.error(f"Erro ao buscar dados para {symbol}: {e}")
            return pd.DataFrame()
    
    def get_multiple_stocks(self, symbols: List[str], period: str = "1y") -> Dict[str, pd.DataFrame]:
        """
        Busca dados para múltiplas ações
        """
        data = {}
        for symbol in symbols:
            stock_data = self.get_stock_data(symbol, period)
            if not stock_data.empty:
                data[symbol] = stock_data
            else:
                logger.warning(f"Nenhum dado encontrado para {symbol}")
        
        logger.info(f"Coletados dados para {len(data)} de {len(symbols)} ações")
        return data
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        """
        Busca preço atual da ação
        """
        try:
            ticker = yf.Ticker(f"{symbol}.SA")
            info = ticker.info
            current_price = info.get('currentPrice') or info.get('regularMarketPrice')
            return float(current_price) if current_price else None
        except Exception as e:
            logger.error(f"Erro ao buscar preço atual para {symbol}: {e}")
            return None
    
    def get_stock_info(self, symbol: str) -> Dict[str, Any]:
        """
        Busca informações detalhadas da ação
        """
        try:
            ticker = yf.Ticker(f"{symbol}.SA")
            info = ticker.info
            
            return {
                'symbol': symbol,
                'company_name': info.get('longName', ''),
                'sector': info.get('sector', ''),
                'industry': info.get('industry', ''),
                'market_cap': info.get('marketCap'),
                'pe_ratio': info.get('trailingPE'),
                'dividend_yield': info.get('dividendYield'),
                'book_value': info.get('bookValue'),
                'price_to_book': info.get('priceToBook'),
                'ebitda': info.get('ebitda'),
                'volume': info.get('volume'),
                'average_volume': info.get('averageVolume')
            }
        except Exception as e:
            logger.error(f"Erro ao buscar informações para {symbol}: {e}")
            return {}
    
    def clear_cache(self):
        """Limpa o cache"""
        self.cache.clear()
        logger.info("Cache limpo")