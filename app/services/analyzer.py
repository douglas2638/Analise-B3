import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
from scipy import stats
import logging
from app.services.data_processor import DataProcessor

logger = logging.getLogger(__name__)

class StockAnalyzer:
    def __init__(self):
        self.processor = DataProcessor()
    
    def _safe_float(self, value, default=0.0):
        """Converte valor para float seguro, tratando NaN e inf"""
        if value is None:
            return default
        try:
            float_value = float(value)
            if np.isnan(float_value) or np.isinf(float_value):
                return default
            return float_value
        except (ValueError, TypeError):
            return default
    
    def _clean_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Limpa um dicionário de valores NaN/inf"""
        if not isinstance(data, dict):
            return {}
        
        cleaned = {}
        for key, value in data.items():
            if isinstance(value, dict):
                cleaned[key] = self._clean_dict(value)
            elif isinstance(value, (list, tuple)):
                cleaned[key] = [self._safe_float(v) if isinstance(v, (int, float)) else v for v in value]
            elif isinstance(value, (int, float)):
                cleaned[key] = self._safe_float(value)
            else:
                cleaned[key] = value
        return cleaned
    
    def analyze_stock(self, symbol: str, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Análise completa de uma ação com tratamento seguro de NaN
        """
        if not self.processor.validate_data(data):
            return {
                'symbol': symbol,
                'error': 'Dados insuficientes para análise',
                'statistics': {},
                'indicators': {},
                'trend_analysis': {},
                'recommendations': ['Dados insuficientes para análise'],
                'score': 50
            }
        
        try:
            # Processa dados com todos os indicadores
            data = self.processor.calculate_all_indicators(data)
            
            # Remove NaN dos dados processados
            data = data.replace([np.inf, -np.inf], np.nan)
            data = data.fillna(method='ffill').fillna(method='bfill')
            
            # Estatísticas básicas
            stats_result = self._calculate_basic_statistics(data)
            
            # Indicadores técnicos
            indicators = self._calculate_technical_indicators(data)
            
            # Análise de tendência
            trend_analysis = self._analyze_trend(data)
            
            # Recomendações
            recommendations = self._generate_recommendations(data, indicators, trend_analysis)
            
            # Score de avaliação (0-100)
            score = self._calculate_stock_score(indicators, trend_analysis, stats_result)
            
            # Garante que todos os valores são serializáveis
            result = {
                'symbol': symbol,
                'statistics': self._clean_dict(stats_result),
                'indicators': self._clean_dict(indicators),
                'trend_analysis': self._clean_dict(trend_analysis),
                'recommendations': recommendations,
                'score': min(100, max(0, self._safe_float(score, 50))),  # Garante entre 0-100
                'timestamp': pd.Timestamp.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Erro na análise de {symbol}: {e}")
            return {
                'symbol': symbol,
                'error': str(e),
                'statistics': {},
                'indicators': {},
                'trend_analysis': {},
                'recommendations': ['Erro na análise'],
                'score': 50
            }
    
    def _calculate_basic_statistics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calcula estatísticas básicas com tratamento de NaN"""
        try:
            # Garante que temos dados válidos
            if data.empty or 'Close' not in data.columns:
                return self._get_default_statistics()
            
            prices = data['Close']
            if len(prices) < 2:
                return self._get_default_statistics()
            
            # Calcula retornos com tratamento de NaN
            returns = data['daily_return'] if 'daily_return' in data.columns else prices.pct_change()
            returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
            
            if len(returns) == 0:
                return self._get_default_statistics()
            
            # Cálculos seguros
            total_return = self._safe_float((prices.iloc[-1] / prices.iloc[0] - 1) * 100, 0.0)
            annual_return = self._safe_float(returns.mean() * 252 * 100, 0.0)
            annual_volatility = self._safe_float(returns.std() * np.sqrt(252) * 100, 0.0)
            
            sharpe_ratio = 0.0
            if returns.std() > 0 and not np.isnan(returns.std()):
                sharpe_ratio = self._safe_float((returns.mean() / returns.std()) * np.sqrt(252), 0.0)
            
            max_drawdown = self._safe_float(self._calculate_max_drawdown(prices), 0.0)
            
            # Volume
            volume_data = data['Volume'] if 'Volume' in data.columns else pd.Series([0] * len(data))
            current_volume = self._safe_float(volume_data.iloc[-1] if len(volume_data) > 0 else 0)
            avg_volume = self._safe_float(volume_data.mean() if len(volume_data) > 0 else 0)
            volume_ratio = self._safe_float(current_volume / avg_volume if avg_volume > 0 else 0.0)
            
            return {
                'price_current': self._safe_float(prices.iloc[-1]),
                'price_variation_1d': self._safe_float(returns.iloc[-1] * 100) if len(returns) > 0 else 0.0,
                'total_return': total_return,
                'annual_return': annual_return,
                'annual_volatility': annual_volatility,
                'sharpe_ratio': max(-10, min(10, sharpe_ratio)),  # Limita Sharpe ratio
                'max_drawdown': max_drawdown,
                'volume': {
                    'current': int(current_volume),
                    'average': int(avg_volume),
                    'ratio': volume_ratio
                }
            }
        except Exception as e:
            logger.error(f"Erro ao calcular estatísticas: {e}")
            return self._get_default_statistics()
    
    def _get_default_statistics(self) -> Dict[str, Any]:
        """Retorna estatísticas padrão em caso de erro"""
        return {
            'price_current': 0.0,
            'price_variation_1d': 0.0,
            'total_return': 0.0,
            'annual_return': 0.0,
            'annual_volatility': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'volume': {
                'current': 0,
                'average': 0,
                'ratio': 0.0
            }
        }
    
    def _calculate_technical_indicators(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calcula indicadores técnicos com tratamento de NaN"""
        try:
            if data.empty:
                return self._get_default_indicators()
            
            latest = data.iloc[-1]
            
            # Médias móveis
            ma_indicators = {}
            for col in data.columns:
                if col.startswith('MA_'):
                    ma_indicators[col] = self._safe_float(latest[col])
            
            # RSI
            rsi_indicators = {}
            for col in data.columns:
                if col.startswith('RSI_'):
                    rsi_value = self._safe_float(latest[col], 50.0)
                    # Limita RSI entre 0-100
                    rsi_indicators[col] = max(0, min(100, rsi_value))
            
            # Bollinger Bands
            bb_position = self._get_bb_position(latest)
            bb_width = self._safe_float(latest.get('BB_width', 0))
            bb_pos_value = self._safe_float(latest.get('BB_position', 0.5))
            
            # MACD
            macd_signal = "NEUTRAL"
            macd_histogram = 0.0
            if 'MACD' in data.columns and 'MACD_signal' in data.columns:
                macd = self._safe_float(latest['MACD'])
                macd_signal_line = self._safe_float(latest['MACD_signal'])
                if macd > macd_signal_line:
                    macd_signal = "BULLISH"
                else:
                    macd_signal = "BEARISH"
                macd_histogram = self._safe_float(latest.get('MACD_histogram', 0))
            
            return {
                'moving_averages': ma_indicators,
                'rsi': rsi_indicators,
                'bollinger_bands': {
                    'position': bb_position,
                    'width': bb_width,
                    'bb_position': bb_pos_value
                },
                'macd': {
                    'signal': macd_signal,
                    'histogram': macd_histogram
                },
                'support_resistance': {
                    'support': self._safe_float(latest.get('support', 0)),
                    'resistance': self._safe_float(latest.get('resistance', 0))
                }
            }
        except Exception as e:
            logger.error(f"Erro ao calcular indicadores: {e}")
            return self._get_default_indicators()
    
    def _get_default_indicators(self) -> Dict[str, Any]:
        """Retorna indicadores padrão em caso de erro"""
        return {
            'moving_averages': {},
            'rsi': {},
            'bollinger_bands': {
                'position': 'UNKNOWN',
                'width': 0.0,
                'bb_position': 0.5
            },
            'macd': {
                'signal': 'NEUTRAL',
                'histogram': 0.0
            },
            'support_resistance': {
                'support': 0.0,
                'resistance': 0.0
            }
        }
    
    def _analyze_trend(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Análise detalhada de tendência com tratamento de NaN"""
        try:
            if data.empty or 'Close' not in data.columns:
                return self._get_default_trend_analysis()
            
            current_price = self._safe_float(data['Close'].iloc[-1])
            
            # Tendência por médias móveis
            ma_trend = self._get_ma_trend(data)
            
            # Tendência por regressão linear (últimos 30 dias)
            prices_30d = data['Close'].tail(30)
            if len(prices_30d) >= 30:
                x = np.arange(len(prices_30d))
                # Remove NaN para regressão
                clean_prices = prices_30d.replace([np.inf, -np.inf], np.nan).dropna()
                if len(clean_prices) >= 30:
                    x_clean = np.arange(len(clean_prices))
                    slope, _, r_value, _, _ = stats.linregress(x_clean, clean_prices)
                    trend_strength = self._safe_float(r_value ** 2)
                    short_trend = "UPTREND" if slope > 0 else "DOWNTREND"
                else:
                    trend_strength = 0.0
                    short_trend = "SIDEWAYS"
            else:
                trend_strength = 0.0
                short_trend = "SIDEWAYS"
            
            return {
                'primary_trend': ma_trend['primary'],
                'secondary_trend': ma_trend['secondary'],
                'short_term_trend': short_trend,
                'trend_strength': trend_strength,
                'key_levels': self._identify_key_levels(data)
            }
        except Exception as e:
            logger.error(f"Erro na análise de tendência: {e}")
            return self._get_default_trend_analysis()
    
    def _get_default_trend_analysis(self) -> Dict[str, Any]:
        """Retorna análise de tendência padrão em caso de erro"""
        return {
            'primary_trend': 'UNKNOWN',
            'secondary_trend': 'UNKNOWN',
            'short_term_trend': 'SIDEWAYS',
            'trend_strength': 0.0,
            'key_levels': {
                'support': 0.0,
                'resistance': 0.0,
                'pivot': 0.0,
                'current_position': 0.5
            }
        }
    
    def _get_ma_trend(self, data: pd.DataFrame) -> Dict[str, str]:
        """Determina tendência baseada em médias móveis"""
        try:
            current = self._safe_float(data['Close'].iloc[-1])
            
            # Verifica se as médias móveis existem
            ma_20 = self._safe_float(data['MA_20'].iloc[-1]) if 'MA_20' in data.columns else current
            ma_50 = self._safe_float(data['MA_50'].iloc[-1]) if 'MA_50' in data.columns else current
            ma_200 = self._safe_float(data['MA_200'].iloc[-1]) if 'MA_200' in data.columns else current
            
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
            
            if recent_data.empty:
                return self._get_default_key_levels()
            
            resistance = self._safe_float(recent_data['High'].max())
            support = self._safe_float(recent_data['Low'].min())
            current_price = self._safe_float(recent_data['Close'].iloc[-1])
            
            pivot = self._safe_float((resistance + support + current_price) / 3)
            
            # Calcula posição atual (0-1)
            if resistance != support:
                current_position = self._safe_float((current_price - support) / (resistance - support))
                # Limita entre 0 e 1
                current_position = max(0, min(1, current_position))
            else:
                current_position = 0.5
            
            return {
                'support': support,
                'resistance': resistance,
                'pivot': pivot,
                'current_position': current_position
            }
        except:
            return self._get_default_key_levels()
    
    def _get_default_key_levels(self) -> Dict[str, float]:
        """Retorna níveis-chave padrão em caso de erro"""
        return {
            'support': 0.0,
            'resistance': 0.0,
            'pivot': 0.0,
            'current_position': 0.5
        }
    
    def _get_bb_position(self, latest: pd.Series) -> str:
        """Posição nas Bandas de Bollinger"""
        try:
            close = self._safe_float(latest['Close'])
            bb_upper = self._safe_float(latest.get('BB_upper', close))
            bb_lower = self._safe_float(latest.get('BB_lower', close))
            bb_middle = self._safe_float(latest.get('BB_middle', close))
            
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
            return self._safe_float(drawdown.min() * 100, 0.0)
        except:
            return 0.0
    
    def _calculate_basic_statistics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calcula estatísticas básicas com tratamento de NaN"""
        try:
            if data.empty or 'Close' not in data.columns:
                return self._get_default_statistics()
            
            prices = data['Close']
            if len(prices) < 2:
                return self._get_default_statistics()
            
            # Calcula retornos
            returns = data['daily_return'] if 'daily_return' in data.columns else prices.pct_change()
            returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
            
            if len(returns) == 0:
                return self._get_default_statistics()
            
            # Cálculos seguros
            total_return = self._safe_float((prices.iloc[-1] / prices.iloc[0] - 1) * 100, 0.0)
            annual_return = self._safe_float(returns.mean() * 252 * 100, 0.0)
            annual_volatility = self._safe_float(returns.std() * np.sqrt(252) * 100, 0.0)
            
            sharpe_ratio = 0.0
            if returns.std() > 0 and not np.isnan(returns.std()):
                sharpe_ratio = self._safe_float((returns.mean() / returns.std()) * np.sqrt(252), 0.0)
            
            max_drawdown = self._safe_float(self._calculate_max_drawdown(prices), 0.0)
            
            # CORREÇÃO: Cálculo de VaR e CVaR
            var_95 = 0.0
            cvar_95 = 0.0
            if len(returns) >= 10:  # Mínimo de dados para cálculo
                try:
                    var_95 = self._safe_float(np.percentile(returns, 5) * 100)
                    # CVaR é a média dos piores 5% dos retornos
                    var_threshold = np.percentile(returns, 5)
                    worst_returns = returns[returns <= var_threshold]
                    if len(worst_returns) > 0:
                        cvar_95 = self._safe_float(worst_returns.mean() * 100)
                except:
                    # Se cálculo falhar, usa valores padrão
                    var_95 = 0.0
                    cvar_95 = 0.0
            
            # Volume
            volume_data = data['Volume'] if 'Volume' in data.columns else pd.Series([0] * len(data))
            current_volume = self._safe_float(volume_data.iloc[-1] if len(volume_data) > 0 else 0)
            avg_volume = self._safe_float(volume_data.mean() if len(volume_data) > 0 else 0)
            volume_ratio = self._safe_float(current_volume / avg_volume if avg_volume > 0 else 0.0)
            
            return {
                'price_current': self._safe_float(prices.iloc[-1]),
                'price_variation_1d': self._safe_float(returns.iloc[-1] * 100) if len(returns) > 0 else 0.0,
                'total_return': total_return,
                'annual_return': annual_return,
                'annual_volatility': annual_volatility,
                'sharpe_ratio': max(-10, min(10, sharpe_ratio)),
                'max_drawdown': max_drawdown,
                'var_95': var_95,  # ← AGORA INCLUÍDO
                'cvar_95': cvar_95,  # ← AGORA INCLUÍDO
                'volume': {
                    'current': int(current_volume),
                    'average': int(avg_volume),
                    'ratio': volume_ratio
                }
            }
        except Exception as e:
            logger.error(f"Erro ao calcular estatísticas: {e}")
            return self._get_default_statistics()
    
    def _get_default_statistics(self) -> Dict[str, Any]:
        return {
            'price_current': 0.0, 
            'price_variation_1d': 0.0, 
            'total_return': 0.0,
            'annual_return': 0.0, 
            'annual_volatility': 0.0, 
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'var_95': 0.0,  # ← INCLUÍDO NO DEFAULT
            'cvar_95': 0.0,  # ← INCLUÍDO NO DEFAULT
            'volume': {
                'current': 0, 
                'average': 0, 
                'ratio': 0.0
            }
        }