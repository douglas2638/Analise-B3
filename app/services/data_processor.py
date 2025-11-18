import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
import logging

logger = logging.getLogger(__name__)

class DataProcessor:
    
    @staticmethod
    def validate_data(data: pd.DataFrame) -> bool:
        """Valida se os dados são adequados para análise"""
        if data.empty:
            return False
        
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in data.columns for col in required_columns):
            return False
            
        if len(data) < 30:  # Mínimo de dados para análise
            return False
            
        return True
    
    @staticmethod
    def calculate_returns(data: pd.DataFrame) -> pd.DataFrame:
        """
        Calcula retornos diários
        """
        if not DataProcessor.validate_data(data):
            return data
            
        try:
            data['daily_return'] = data['Close'].pct_change()
            data['log_return'] = np.log(data['Close'] / data['Close'].shift(1))
            data['cumulative_return'] = (1 + data['daily_return']).cumprod() - 1
            return data
        except Exception as e:
            logger.error(f"Erro ao calcular retornos: {e}")
            return data
    
    @staticmethod
    def calculate_volatility(data: pd.DataFrame, windows: List[int] = [30, 60, 90]) -> pd.DataFrame:
        """
        Calcula volatilidade móvel para diferentes períodos
        """
        if not DataProcessor.validate_data(data):
            return data
            
        try:
            for window in windows:
                data[f'volatility_{window}d'] = data['daily_return'].rolling(window=window).std() * np.sqrt(252)
            return data
        except Exception as e:
            logger.error(f"Erro ao calcular volatilidade: {e}")
            return data
    
    @staticmethod
    def calculate_moving_averages(data: pd.DataFrame) -> pd.DataFrame:
        """
        Calcula médias móveis
        """
        if not DataProcessor.validate_data(data):
            return data
            
        try:
            periods = [5, 10, 20, 50, 200]
            for period in periods:
                data[f'MA_{period}'] = data['Close'].rolling(window=period).mean()
            return data
        except Exception as e:
            logger.error(f"Erro ao calcular médias móveis: {e}")
            return data
    
    @staticmethod
    def calculate_rsi(data: pd.DataFrame, windows: List[int] = [14, 21]) -> pd.DataFrame:
        """
        Calcula RSI (Relative Strength Index) para diferentes períodos
        """
        if not DataProcessor.validate_data(data):
            return data
            
        try:
            for window in windows:
                delta = data['Close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
                rs = gain / loss
                data[f'RSI_{window}'] = 100 - (100 / (1 + rs))
            return data
        except Exception as e:
            logger.error(f"Erro ao calcular RSI: {e}")
            return data
    
    @staticmethod
    def calculate_bollinger_bands(data: pd.DataFrame, window: int = 20, num_std: int = 2) -> pd.DataFrame:
        """
        Calcula Bandas de Bollinger
        """
        if not DataProcessor.validate_data(data):
            return data
            
        try:
            data['BB_middle'] = data['Close'].rolling(window=window).mean()
            bb_std = data['Close'].rolling(window=window).std()
            data['BB_upper'] = data['BB_middle'] + (bb_std * num_std)
            data['BB_lower'] = data['BB_middle'] - (bb_std * num_std)
            data['BB_width'] = (data['BB_upper'] - data['BB_lower']) / data['BB_middle']
            data['BB_position'] = (data['Close'] - data['BB_lower']) / (data['BB_upper'] - data['BB_lower'])
            return data
        except Exception as e:
            logger.error(f"Erro ao calcular Bollinger Bands: {e}")
            return data
    
    @staticmethod
    def calculate_macd(data: pd.DataFrame) -> pd.DataFrame:
        """
        Calcula MACD (Moving Average Convergence Divergence)
        """
        if not DataProcessor.validate_data(data):
            return data
            
        try:
            exp1 = data['Close'].ewm(span=12).mean()
            exp2 = data['Close'].ewm(span=26).mean()
            data['MACD'] = exp1 - exp2
            data['MACD_signal'] = data['MACD'].ewm(span=9).mean()
            data['MACD_histogram'] = data['MACD'] - data['MACD_signal']
            return data
        except Exception as e:
            logger.error(f"Erro ao calcular MACD: {e}")
            return data
    
    @staticmethod
    def calculate_support_resistance(data: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        """
        Identifica níveis de suporte e resistência
        """
        if not DataProcessor.validate_data(data):
            return data
            
        try:
            data['resistance'] = data['High'].rolling(window=window).max()
            data['support'] = data['Low'].rolling(window=window).min()
            return data
        except Exception as e:
            logger.error(f"Erro ao calcular suporte/resistência: {e}")
            return data
    
    @staticmethod
    def calculate_all_indicators(data: pd.DataFrame) -> pd.DataFrame:
        """
        Calcula todos os indicadores técnicos
        """
        if not DataProcessor.validate_data(data):
            return data
            
        data = DataProcessor.calculate_returns(data)
        data = DataProcessor.calculate_volatility(data)
        data = DataProcessor.calculate_moving_averages(data)
        data = DataProcessor.calculate_rsi(data)
        data = DataProcessor.calculate_bollinger_bands(data)
        data = DataProcessor.calculate_macd(data)
        data = DataProcessor.calculate_support_resistance(data)
        
        return data