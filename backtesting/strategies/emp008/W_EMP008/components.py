import logging
import pandas as pd
from functools import cached_property
from typing import Optional, Dict, Any
from W_EMP008.preprocess import PPFactors, FactorPreProcess
from W_EMP008.covar import CovarianceMatrix
from W_EMP008.residuals import Residuals
from W_EMP008.regression import MultiFactorRegression


class RiskModelComponents:
    def __init__(self, 
                 data_dir: str, 
                 use_excel_spec: bool = True, 
                 window: int = 36):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        if not self.logger.hasHandlers():
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        self.data_dir = data_dir
        self.use_excel_spec = use_excel_spec
        self.window = window
        
        self._cache: Dict[str, Any] = {}
    
    @cached_property
    def pp_factors(self) -> PPFactors:
        self.logger.info("pp_factors property accessed")
        return FactorPreProcess(self.data_dir).pp_factors
    
    @cached_property
    def covariance_matrix(self) -> CovarianceMatrix:
        self.logger.info("covariance_matrix property accessed")
        return CovarianceMatrix(self.data_dir, use_excel_spec=self.use_excel_spec)
    
    @cached_property
    def residuals(self) -> Residuals:
        self.logger.info("residuals property accessed")
        return Residuals(self.data_dir, use_excel_spec=self.use_excel_spec)
    
    @cached_property
    def bm(self) -> pd.DataFrame:
        self.logger.info("bm property accessed")
        df = self.covariance_matrix._get_preprocessed_data("BM")
        return df.div(df.sum(axis=0), axis=1)
    
    @cached_property
    def available_dates(self) -> list[str]:
        self.logger.info("available_dates property accessed")
        return list(self.pp_factors.keys())
    
    @cached_property
    def regression_dates(self) -> list[pd.Timestamp]:
        self.logger.info("regression_dates property accessed")
        return self.covariance_matrix.regression_dates
    
    @cached_property
    def tickers(self) -> list[str]:
        self.logger.info("tickers property accessed")
        return self.covariance_matrix.tickers
    
    def _get_cache_key(self, method: str, **kwargs) -> str:
        key_parts = [method] + [f"{k}={v}" for k, v in sorted(kwargs.items()) if v is not None]
        return "_".join(key_parts)
    
    def factors(self, date: str) -> pd.DataFrame:
        self.logger.info(f"factors called with date={date}")
        cache_key = self._get_cache_key("factors", date=date)
        if cache_key not in self._cache:
            self._cache[cache_key] = self.pp_factors[date]
        return self._cache[cache_key]
    
    def factor_covariance(self, date: str, periods: Optional[int] = None) -> pd.DataFrame:
        self.logger.info(f"factor_covariance called with date={date}, periods={periods}")
        periods = periods or self.window
        cache_key = self._get_cache_key("factor_cov", date=date, periods=periods)
        
        if cache_key not in self._cache:
            self._cache[cache_key] = self.covariance_matrix.get_matrix(date, periods=periods)
        return self._cache[cache_key]
    
    def residual_covariance(self, end_date: Optional[str] = None, window: Optional[int] = None) -> pd.DataFrame:
        self.logger.info(f"residual_covariance called with end_date={end_date}, window={window}")
        window = window or self.window
        cache_key = self._get_cache_key("residual_cov", end_date=end_date, window=window)
        
        if cache_key not in self._cache:
            self._cache[cache_key] = self.residuals.get_residual_covariance_matrix(
                window=window, end_date=end_date
            )
        return self._cache[cache_key]
    
    @property
    def coef(self) -> pd.DataFrame:
        return MultiFactorRegression(self.data_dir, use_excel_spec=self.use_excel_spec).coefficients
    
    def expected_alpha(self, end_date: Optional[str] = None, window: int = 36) -> pd.Series:
        self.logger.info(f"expected_alpha called with end_date={end_date}, window={window}")
        cache_key = self._get_cache_key("expected_alpha", end_date=end_date, window=window)
        if cache_key in self._cache:
            return self._cache[cache_key]
        coef_data = self.coef.copy()
        if end_date:
            end_timestamp = pd.to_datetime(end_date)
            mask = coef_data.index <= end_timestamp
            coef_data = coef_data[mask]
        recent_data = coef_data.tail(window)
        alpha_values = {}
        for factor in recent_data.columns:
            mean_val = recent_data[factor].mean()
            if factor in ['DY', 'Momentum_12M']:
                alpha_values[factor] = mean_val if mean_val >= 0 else 0.0
            elif factor == 'LnMktcap':
                alpha_values[factor] = mean_val if mean_val <= 0 else 0.0
            else:
                alpha_values[factor] = mean_val
        res = pd.Series(alpha_values)
        self._cache[cache_key] = res
        return res

    def clear_cache(self):
        self._cache.clear()
        cached_props = ['pp_factors', 'covariance_matrix', 'residuals', 'bm', 
                       'available_dates', 'regression_dates', 'tickers']
        for attr in cached_props:
            if hasattr(self, f'_{attr}'):
                delattr(self, f'_{attr}')
    
    @property 
    def cache_info(self) -> Dict[str, int]:
        cache_types = {
            "factors": "factors",
            "factor_cov": "factor_cov", 
            "residual_cov": "residual_cov",
        }
        
        info = {"total_cached_items": len(self._cache)}
        for cache_type, prefix in cache_types.items():
            info[f"{cache_type}_cache"] = len([k for k in self._cache.keys() if k.startswith(prefix)])
        return info
    
    def __repr__(self) -> str:
        return f"RiskModelComponents(data_dir='{self.data_dir}', use_excel_spec={self.use_excel_spec}, window={self.window})"
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.clear_cache()


if __name__ == "__main__":
    risk_model = RiskModelComponents("W_EMP008/DATA", use_excel_spec=True, window=36)

    factors = risk_model.factors('20250331')
    factor_cov = risk_model.factor_covariance('20250228', 36)
    residual_cov = risk_model.residual_covariance(end_date="20250331")
    expected_alpha = risk_model.expected_alpha(end_date="20250228", window=36)
