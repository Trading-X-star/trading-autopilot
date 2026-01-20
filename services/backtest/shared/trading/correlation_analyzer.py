"""
Portfolio Correlation Analyzer
Анализ корреляций между позициями для диверсификации
"""
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

@dataclass
class CorrelationConfig:
    lookback_period: int = 60  # Дней для расчёта корреляции
    high_correlation_threshold: float = 0.7
    max_correlated_exposure: float = 0.25  # Макс 25% в сильно коррелированных активах
    min_data_points: int = 20

@dataclass
class CorrelationResult:
    symbol1: str
    symbol2: str
    correlation: float
    is_high: bool
    rolling_correlation: Optional[np.ndarray] = None

@dataclass
class PortfolioCorrelationAnalysis:
    correlation_matrix: Dict[str, Dict[str, float]]
    high_correlations: List[CorrelationResult]
    average_correlation: float
    diversification_ratio: float
    concentration_risk: float
    recommendations: List[str]

class CorrelationAnalyzer:
    def __init__(self, config: Optional[CorrelationConfig] = None):
        self.config = config or CorrelationConfig()
        self._price_data: Dict[str, np.ndarray] = {}
    
    def update_prices(self, symbol: str, prices: np.ndarray):
        """Обновить ценовые данные для символа"""
        self._price_data[symbol] = prices
    
    def calculate_returns(self, prices: np.ndarray) -> np.ndarray:
        """Рассчитать логарифмические доходности"""
        if len(prices) < 2:
            return np.array([])
        return np.diff(np.log(prices))
    
    def calculate_correlation(self, symbol1: str, symbol2: str) -> CorrelationResult:
        """Рассчитать корреляцию между двумя активами"""
        
        if symbol1 not in self._price_data or symbol2 not in self._price_data:
            return CorrelationResult(symbol1, symbol2, 0.0, False)
        
        prices1 = self._price_data[symbol1]
        prices2 = self._price_data[symbol2]
        
        # Выровнять длину
        min_len = min(len(prices1), len(prices2))
        if min_len < self.config.min_data_points:
            return CorrelationResult(symbol1, symbol2, 0.0, False)
        
        returns1 = self.calculate_returns(prices1[-min_len:])
        returns2 = self.calculate_returns(prices2[-min_len:])
        
        # Pearson correlation
        correlation = np.corrcoef(returns1, returns2)[3]
        
        if np.isnan(correlation):
            correlation = 0.0
        
        is_high = abs(correlation) >= self.config.high_correlation_threshold
        
        return CorrelationResult(
            symbol1=symbol1,
            symbol2=symbol2,
            correlation=float(correlation),
            is_high=is_high
        )
    
    def calculate_correlation_matrix(self, symbols: List[str]) -> Dict[str, Dict[str, float]]:
        """Рассчитать матрицу корреляций"""
        matrix = {}
        
        for s1 in symbols:
            matrix[s1] = {}
            for s2 in symbols:
                if s1 == s2:
                    matrix[s1][s2] = 1.0
                elif s2 in matrix and s1 in matrix[s2]:
                    matrix[s1][s2] = matrix[s2][s1]
                else:
                    result = self.calculate_correlation(s1, s2)
                    matrix[s1][s2] = result.correlation
        
        return matrix
    
    def calculate_diversification_ratio(
        self,
        symbols: List[str],
        weights: Dict[str, float]
    ) -> float:
        """
        Diversification Ratio = weighted avg volatility / portfolio volatility
        DR > 1 означает диверсификацию
        """
        if not symbols or not weights:
            return 1.0
        
        volatilities = {}
        for symbol in symbols:
            if symbol in self._price_data:
                returns = self.calculate_returns(self._price_data[symbol])
                if len(returns) > 0:
                    volatilities[symbol] = np.std(returns) * np.sqrt(252)
                else:
                    volatilities[symbol] = 0.2
            else:
                volatilities[symbol] = 0.2
        
        # Weighted average volatility
        weighted_vol = sum(
            weights.get(s, 0) * volatilities.get(s, 0.2)
            for s in symbols
        )
        
        # Portfolio volatility
        corr_matrix = self.calculate_correlation_matrix(symbols)
        
        portfolio_var = 0
        for s1 in symbols:
            for s2 in symbols:
                w1 = weights.get(s1, 0)
                w2 = weights.get(s2, 0)
                v1 = volatilities.get(s1, 0.2)
                v2 = volatilities.get(s2, 0.2)
                corr = corr_matrix.get(s1, {}).get(s2, 0)
                portfolio_var += w1 * w2 * v1 * v2 * corr
        
        portfolio_vol = np.sqrt(max(0, portfolio_var))
        
        if portfolio_vol == 0:
            return 1.0
        
        return weighted_vol / portfolio_vol
    
    def analyze_portfolio(
        self,
        positions: Dict[str, float],  # symbol -> position_value
        portfolio_value: float
    ) -> PortfolioCorrelationAnalysis:
        """Полный анализ корреляций портфеля"""
        
        symbols = list(positions.keys())
        
        if len(symbols) < 2:
            return PortfolioCorrelationAnalysis(
                correlation_matrix={},
                high_correlations=[],
                average_correlation=0.0,
                diversification_ratio=1.0,
                concentration_risk=1.0 if len(symbols) == 1 else 0.0,
                recommendations=["Добавьте больше позиций для диверсификации"]
            )
        
        # Calculate weights
        weights = {s: v / portfolio_value for s, v in positions.items()}
        
        # Correlation matrix
        corr_matrix = self.calculate_correlation_matrix(symbols)
        
        # Find high correlations
        high_correlations = []
        all_correlations = []
        
        for i, s1 in enumerate(symbols):
            for s2 in symbols[i+1:]:
                corr = corr_matrix[s1][s2]
                all_correlations.append(abs(corr))
                
                if abs(corr) >= self.config.high_correlation_threshold:
                    high_correlations.append(CorrelationResult(
                        symbol1=s1,
                        symbol2=s2,
                        correlation=corr,
                        is_high=True
                    ))
        
        # Average correlation
        avg_corr = np.mean(all_correlations) if all_correlations else 0.0
        
        # Diversification ratio
        div_ratio = self.calculate_diversification_ratio(symbols, weights)
        
        # Concentration risk (Herfindahl index)
        concentration = sum(w ** 2 for w in weights.values())
        
        # Recommendations
        recommendations = []
        
        if avg_corr > 0.5:
            recommendations.append("⚠️ Высокая средняя корреляция портфеля. Рассмотрите добавление некоррелированных активов.")
        
        for hc in high_correlations:
            combined_weight = weights.get(hc.symbol1, 0) + weights.get(hc.symbol2, 0)
            if combined_weight > self.config.max_correlated_exposure:
                recommendations.append(
                    f"⚠️ {hc.symbol1} и {hc.symbol2} сильно коррелированы ({hc.correlation:.2f}) "
                    f"с общей долей {combined_weight:.1%}. Рекомендуется снизить до {self.config.max_correlated_exposure:.0%}."
                )
        
        if concentration > 0.3:
            recommendations.append("⚠️ Высокая концентрация портфеля. Рассмотрите более равномерное распределение.")
        
        if div_ratio < 1.2:
            recommendations.append("ℹ️ Низкий коэффициент диверсификации. Портфель может быть недостаточно диверсифицирован.")
        
        if not recommendations:
            recommendations.append("✅ Корреляционный профиль портфеля в норме.")
        
        logger.info(f"Portfolio Analysis: avg_corr={avg_corr:.2f}, div_ratio={div_ratio:.2f}, concentration={concentration:.2f}")
        
        return PortfolioCorrelationAnalysis(
            correlation_matrix=corr_matrix,
            high_correlations=high_correlations,
            average_correlation=float(avg_corr),
            diversification_ratio=float(div_ratio),
            concentration_risk=float(concentration),
            recommendations=recommendations
        )
    
    def should_allow_trade(
        self,
        new_symbol: str,
        new_value: float,
        current_positions: Dict[str, float],
        portfolio_value: float
    ) -> Tuple[bool, str]:
        """Проверить, допустима ли новая сделка с точки зрения корреляции"""
        
        if new_symbol not in self._price_data:
            return True, "Нет данных для анализа корреляции"
        
        for symbol, value in current_positions.items():
            if symbol == new_symbol:
                continue
            
            if symbol not in self._price_data:
                continue
            
            result = self.calculate_correlation(new_symbol, symbol)
            
            if result.is_high:
                combined_exposure = (value + new_value) / portfolio_value
                
                if combined_exposure > self.config.max_correlated_exposure:
                    return False, (
                        f"Отклонено: {new_symbol} сильно коррелирован с {symbol} "
                        f"(r={result.correlation:.2f}). Общая экспозиция {combined_exposure:.1%} "
                        f"превышает лимит {self.config.max_correlated_exposure:.0%}"
                    )
        
        return True, "OK"

# Singleton
correlation_analyzer = CorrelationAnalyzer()
