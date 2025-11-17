import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
from scipy import stats
import logging
from app.services.data_processor import DataProcessor

logger = logging.getLogger(__name__)

class StockAnalyzer:
    def __init__(self, data_processor: DataProcessor):
        self.processor = data_processor
    
    def analyze_stock(self, symbol: str, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Análise completa de uma ação
        """
        if not self.processor.validate_data(data):
            return {
                'symbol': symbol,
                'error': 'Dados insuficientes para análise',
                'statistics': {},
                'indicators': {},
                'recommendations': ['Dados insuficientes para análise']
            }
        
        try:
            # Processa dados com todos os indicadores
            data = self.processor.calculate_all_indicators(data)
            
            # Estatísticas básicas
            stats = self._calculate_basic_statistics(data)
            
            # Indicadores técnicos
            indicators = self._calculate_technical_indicators(data)
            
            # Análise de tendência
            trend_analysis = self._analyze_trend(data)
            
            # Recomendações
            recommendations = self._generate_recommendations(data, indicators, trend_analysis)
            
            # Score de avaliação (0-100)
            score = self._calculate_stock_score(indicators, trend_analysis, stats)
            
            return {
                'symbol': symbol,
                'statistics': stats,
                'indicators': indicators,
                'trend_analysis': trend_analysis,
                'recommendations': recommendations,
                'score': score,
                'timestamp': pd.Timestamp.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Erro na análise de {symbol}: {e}")
            return {
                'symbol': symbol,
                'error': str(e),
                'statistics': {},
                'indicators': {},
                'recommendations': ['Erro na análise']
            }
    
    def _calculate_basic_statistics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calcula estatísticas básicas"""
        try:
            returns = data['daily_return'].dropna()
            prices = data['Close']
            
            # Retornos
            total_return = (prices.iloc[-1] / prices.iloc[0] - 1) * 100
            annual_return = returns.mean() * 252 * 100
            annual_volatility = returns.std() * np.sqrt(252) * 100
            sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
            
            # Risco
            max_drawdown = self._calculate_max_drawdown(prices)
            var_95 = np.percentile(returns, 5) * 100
            cvar_95 = returns[returns <= returns.quantile(0.05)].mean() * 100
            
            # Volume
            volume_stats = {
                'current': int(data['Volume'].iloc[-1]),
                'average': int(data['Volume'].mean()),
                'ratio': data['Volume'].iloc[-1] / data['Volume'].mean() if data['Volume'].mean() > 0 else 0
            }
            
            return {
                'price_current': float(prices.iloc[-1]),
                'price_variation_1d': float(returns.iloc[-1] * 100) if len(returns) > 0 else 0,
                'total_return': float(total_return),
                'annual_return': float(annual_return),
                'annual_volatility': float(annual_volatility),
                'sharpe_ratio': float(sharpe_ratio),
                'max_drawdown': float(max_drawdown),
                'var_95': float(var_95),
                'cvar_95': float(cvar_95),
                'volume': volume_stats
            }
        except Exception as e:
            logger.error(f"Erro ao calcular estatísticas: {e}")
            return {}
    
    def _calculate_technical_indicators(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calcula indicadores técnicos"""
        try:
            latest = data.iloc[-1]
            
            # Médias móveis
            ma_indicators = {}
            for col in data.columns:
                if col.startswith('MA_'):
                    ma_indicators[col] = float(latest[col])
            
            # RSI
            rsi_indicators = {}
            for col in data.columns:
                if col.startswith('RSI_'):
                    rsi_indicators[col] = float(latest[col])
            
            # Bollinger Bands
            bb_position = self._get_bb_position(latest)
            
            # MACD
            macd_signal = "NEUTRAL"
            if 'MACD' in data.columns and 'MACD_signal' in data.columns:
                macd = latest['MACD']
                macd_signal_line = latest['MACD_signal']
                if macd > macd_signal_line:
                    macd_signal = "BULLISH"
                else:
                    macd_signal = "BEARISH"
            
            return {
                'moving_averages': ma_indicators,
                'rsi': rsi_indicators,
                'bollinger_bands': {
                    'position': bb_position,
                    'width': float(latest.get('BB_width', 0)),
                    'bb_position': float(latest.get('BB_position', 0))
                },
                'macd': {
                    'signal': macd_signal,
                    'histogram': float(latest.get('MACD_histogram', 0))
                },
                'support_resistance': {
                    'support': float(latest.get('support', 0)),
                    'resistance': float(latest.get('resistance', 0))
                }
            }
        except Exception as e:
            logger.error(f"Erro ao calcular indicadores: {e}")
            return {}
    
    def _analyze_trend(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Análise detalhada de tendência"""
        try:
            current_price = data['Close'].iloc[-1]
            
            # Tendência por médias móveis
            ma_trend = self._get_ma_trend(data)
            
            # Tendência por regressão linear (últimos 30 dias)
            prices_30d = data['Close'].tail(30)
            if len(prices_30d) >= 30:
                x = np.arange(len(prices_30d))
                slope, _, r_value, _, _ = stats.linregress(x, prices_30d)
                trend_strength = r_value ** 2
                short_trend = "UPTREND" if slope > 0 else "DOWNTREND"
            else:
                trend_strength = 0
                short_trend = "SIDEWAYS"
            
            return {
                'primary_trend': ma_trend['primary'],
                'secondary_trend': ma_trend['secondary'],
                'short_term_trend': short_trend,
                'trend_strength': float(trend_strength),
                'key_levels': self._identify_key_levels(data)
            }
        except Exception as e:
            logger.error(f"Erro na análise de tendência: {e}")
            return {}
    
    def _get_ma_trend(self, data: pd.DataFrame) -> Dict[str, str]:
        """Determina tendência baseada em médias móveis"""
        try:
            current = data['Close'].iloc[-1]
            ma_20 = data['MA_20'].iloc[-1] if 'MA_20' in data.columns else current
            ma_50 = data['MA_50'].iloc[-1] if 'MA_50' in data.columns else current
            ma_200 = data['MA_200'].iloc[-1] if 'MA_200' in data.columns else current
            
            # Tendência primária (MA200)
            if current > ma_200 and ma_50 > ma_200:
                primary = "BULL_MARKET"
            elif current < ma_200 and ma_50 < ma_200:
                primary = "BEAR_MARKET"
            else:
                primary = "TRANSITION"
            
            # Tendência secundária (MA50)
            if current > ma_50 and ma_20 > ma_50:
                secondary = "UPTREND"
            elif current < ma_50 and ma_20 < ma_50:
                secondary = "DOWNTREND"
            else:
                secondary = "CONSOLIDATION"
            
            return {'primary': primary, 'secondary': secondary}
        except:
            return {'primary': 'UNKNOWN', 'secondary': 'UNKNOWN'}
    
    def _identify_key_levels(self, data: pd.DataFrame) -> Dict[str, float]:
        """Identifica níveis-chave de suporte e resistência"""
        try:
            # Usa os últimos 60 dias para identificar níveis
            recent_data = data.tail(60)
            
            resistance = recent_data['High'].max()
            support = recent_data['Low'].min()
            pivot = (resistance + support + recent_data['Close'].iloc[-1]) / 3
            
            return {
                'support': float(support),
                'resistance': float(resistance),
                'pivot': float(pivot),
                'current_position': float((recent_data['Close'].iloc[-1] - support) / (resistance - support))
            }
        except:
            return {}
    
    def _get_bb_position(self, latest: pd.Series) -> str:
        """Posição nas Bandas de Bollinger"""
        try:
            close = latest['Close']
            bb_upper = latest.get('BB_upper', close)
            bb_lower = latest.get('BB_lower', close)
            bb_middle = latest.get('BB_middle', close)
            
            if close >= bb_upper:
                return 'OVERBOUGHT'
            elif close <= bb_lower:
                return 'OVERSOLD'
            elif abs(close - bb_upper) < abs(close - bb_middle):
                return 'UPPER_BAND'
            elif abs(close - bb_lower) < abs(close - bb_middle):
                return 'LOWER_BAND'
            else:
                return 'MIDDLE_BAND'
        except:
            return 'UNKNOWN'
    
    def _generate_recommendations(self, data: pd.DataFrame, indicators: Dict[str, Any], trend_analysis: Dict[str, Any]) -> List[str]:
        """Gera recomendações baseadas na análise"""
        recommendations = []
        
        try:
            # Análise RSI
            rsi_14 = indicators.get('rsi', {}).get('RSI_14', 50)
            if rsi_14 > 80:
                recommendations.append("🔴 RSI EXTREMAMENTE SOBRECOMPRADO - Alto risco de correção")
            elif rsi_14 > 70:
                recommendations.append("🟡 RSI indica sobrecompra - Considere tomar lucros")
            elif rsi_14 < 20:
                recommendations.append("🟢 RSI EXTREMAMENTE SOBREVENDIDO - Oportunidade de compra")
            elif rsi_14 < 30:
                recommendations.append("🟢 RSI indica sobrevenda - Possível oportunidade")
            
            # Análise de tendência
            trend = trend_analysis.get('primary', '')
            if trend == 'BULL_MARKET':
                recommendations.append("📈 Mercado em tendência de alta - Bom para posições longas")
            elif trend == 'BEAR_MARKET':
                recommendations.append("📉 Mercado em tendência de baixa - Cuidado com novas posições")
            
            # Análise Bandas de Bollinger
            bb_position = indicators.get('bollinger_bands', {}).get('position', '')
            if bb_position == 'OVERBOUGHT':
                recommendations.append("⚠️  Preço próximo à banda superior - Possível correção")
            elif bb_position == 'OVERSOLD':
                recommendations.append("💡 Preço próximo à banda inferior - Possível recuperação")
            
            # Análise de volume
            volume_ratio = data['statistics'].get('volume', {}).get('ratio', 1) if 'statistics' in data else 1
            if volume_ratio > 2:
                recommendations.append("🔥 Volume acima da média - Confirma movimento")
            elif volume_ratio < 0.5:
                recommendations.append("💤 Volume abaixo da média - Movimento pouco confiável")
                
        except Exception as e:
            logger.error(f"Erro ao gerar recomendações: {e}")
            recommendations.append("⚠️  Erro na geração de recomendações")
        
        return recommendations if recommendations else ["📊 Análise neutra - Aguardar melhores oportunidades"]
    
    def _calculate_stock_score(self, indicators: Dict[str, Any], trend_analysis: Dict[str, Any], statistics: Dict[str, Any]) -> int:
        """Calcula score de 0-100 para a ação"""
        score = 50  # Base
        
        try:
            # Fator RSI
            rsi_14 = indicators.get('rsi', {}).get('RSI_14', 50)
            if 30 <= rsi_14 <= 70:
                score += 10
            elif 40 <= rsi_14 <= 60:
                score += 20
            
            # Fator Tendência
            trend = trend_analysis.get('primary', '')
            if trend == 'BULL_MARKET':
                score += 15
            elif trend == 'BEAR_MARKET':
                score -= 15
            
            # Fator Volatilidade
            volatility = statistics.get('annual_volatility', 0)
            if 10 <= volatility <= 30:
                score += 10
            elif volatility > 50:
                score -= 10
            
            # Fator Sharpe
            sharpe = statistics.get('sharpe_ratio', 0)
            if sharpe > 1:
                score += 10
            elif sharpe < 0:
                score -= 10
            
            return max(0, min(100, score))
            
        except:
            return 50
    
    def _calculate_max_drawdown(self, prices: pd.Series) -> float:
        """Calcula máximo drawdown"""
        try:
            cumulative_max = prices.cummax()
            drawdown = (prices - cumulative_max) / cumulative_max
            return float(drawdown.min() * 100)
        except:
            return 0.0