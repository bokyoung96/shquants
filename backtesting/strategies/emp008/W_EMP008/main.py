import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Any
from W_EMP008.optimize import PortfolioOptimizer
from W_EMP008.report import ReportGenerator


"""
Data Processing Pipeline:

┌─────────────┐    ┌──────────────┐    ┌─────────────┐    ┌─────────────┐
│   loader    │───▶│  preprocess  │───▶│ regression  │───▶│    covar    │
│   데이터 저장   │    │   데이터 전처리  │    │ 회귀계수 산출 │    │ 공분산 산출   │
└─────────────┘    └──────────────┘    └─────────────┘    └─────────────┘
                                                              │
┌─────────────┐    ┌──────────────┐    ┌─────────────┐       │
│  optimize   │◀───│  components  │◀───│  residuals  │◀──────┘
│   최적화     │    │   항목 정리    │    │  잔차 산출   │
└─────────────┘    └──────────────┘    └─────────────┘

Portfolio Optimization:

    Objective:  maximize alpha exposure
                    αᵗ · (Zᵗx)

    Constraints:
    ┌─────────────────────────────────────┐
    │   Aeq @ x = beq                     │   ← 선형 등식 제약
    └─────────────────────────────────────┘
    ┌─────────────────────────────────────┐
    │   Tracking Error constraint         │   ← 비선형 부등식 제약
    └─────────────────────────────────────┘
    ┌─────────────────────────────────────┐
    │   bounds: x ≥ -w                    │   ← 각 종목 비중 ≥ 0
    └─────────────────────────────────────┘
"""




class Runner:
    def __init__(self, 
                 data_dir: str = "W_EMP008/DATA",
                 use_excel_spec: bool = True,
                 window: int = 36,
                 tracking_error: float = 0.7/np.sqrt(12)):
        self.optimizer = PortfolioOptimizer(data_dir, use_excel_spec, window, tracking_error)
        self.result = None
        self.cache = {}
        self.last_date1 = None
        self.last_date2 = None
        
    def factors(self, date: str) -> pd.DataFrame:
        key = f"factors_{date}"
        if key not in self.cache:
            self.cache[key] = self.optimizer.factors(date)
        factors = self.cache[key]
        print(f"Factors: {factors.shape} - {list(factors.columns)}")
        return factors
    
    def covariance(self, date: str) -> pd.DataFrame:
        key = f"covariance_{date}"
        if key not in self.cache:
            self.cache[key] = self.optimizer.factor_covariance(date, self.optimizer.window)
        factor_cov = self.cache[key]
        print(f"Factor covariance: {factor_cov.shape}")
        return factor_cov
    
    def residual_cov(self, date: str) -> pd.DataFrame:
        key = f"residual_cov_{date}"
        if key not in self.cache:
            self.cache[key] = self.optimizer.residual_covariance(end_date=date)
        residual_cov = self.cache[key]
        diag_mean = np.diag(residual_cov.values).mean()
        print(f"Residual covariance: {residual_cov.shape}, diag mean: {diag_mean:.6f}")
        return residual_cov
    
    def e_alpha(self, date: str) -> pd.Series:
        key = f"alpha_{date}"
        if key not in self.cache:
            self.cache[key] = self.optimizer.expected_alpha(end_date=date, window=self.optimizer.window)
        alpha_exp = self.cache[key]
        print(f"Alpha: {alpha_exp.shape}, mean: {alpha_exp.mean():.6f}, std: {alpha_exp.std():.6f}")
        return alpha_exp
    
    def full_covariance(self, date1: str, date2: str) -> np.ndarray:
        key = f"full_cov_{date1}_{date2}"
        if key not in self.cache:
            factors = self.factors(date2)
            factor_cov = self.covariance(date1)
            residual_cov = self.residual_cov(date2)
            
            common_tickers = factors.index.intersection(residual_cov.index)
            common_factors = factors.columns.intersection(factor_cov.index)
            
            factors_aligned = factors.loc[common_tickers, common_factors]
            factor_cov_aligned = factor_cov.loc[common_factors, common_factors]
            residual_cov_aligned = residual_cov.loc[common_tickers, common_tickers]
            
            self.cache[key] = self.optimizer._build_covariance_matrix(factor_cov_aligned, factors_aligned, residual_cov_aligned)
        
        M = self.cache[key]
        eigenvals = np.linalg.eigvals(M)
        print(f"Full covariance: {M.shape}, condition: {np.linalg.cond(M):.2e}, negative eigenvals: {(eigenvals < 0).sum()}")
        return M
    
    @property
    def current_factors(self) -> pd.DataFrame:
        if self.last_date2 is None:
            raise ValueError("Run optimization first")
        return self.factors(self.last_date2)
    
    @property
    def current_covariance(self) -> pd.DataFrame:
        if self.last_date1 is None:
            raise ValueError("Run optimization first")
        return self.covariance(self.last_date1)
    
    @property
    def current_residual_cov(self) -> pd.DataFrame:
        if self.last_date2 is None:
            raise ValueError("Run optimization first")
        return self.residual_cov(self.last_date2)
    
    @property
    def current_alpha(self) -> pd.Series:
        if self.last_date1 is None:
            raise ValueError("Run optimization first")
        return self.e_alpha(self.last_date1)
    
    @property
    def current_full_covariance(self) -> np.ndarray:
        if self.last_date1 is None or self.last_date2 is None:
            raise ValueError("Run optimization first")
        return self.full_covariance(self.last_date1, self.last_date2)
    
    def run(self,
            date1: str,
            date2: str,
            individual_limits: Optional[Dict[int, float]] = None,
            exclude_stocks: Optional[List[int]] = None,
            check_inputs: bool = True) -> Dict[str, Any]:
        
        self.last_date1 = date1
        self.last_date2 = date2
        
        if check_inputs:
            self.factors(date2)
            self.covariance(date1)
            self.residual_cov(date2)
            self.e_alpha(date1)
            self.full_covariance(date1, date2)
            print()
        
        self.result = self.optimizer.optimize_portfolio(
            date1=date1,
            date2=date2,
            individual_limits=individual_limits,
            exclude_stocks=exclude_stocks
        )
        
        print(f"Success: {self.result['success']}")
        print(f"Objective: {self.result['objective_value']:.6f}")
        print(f"Tracking Error: {self.result['tracking_error']:.6f}")
        
        active_weights = self.result['active_weights']
        significant = active_weights[abs(active_weights) > 0.001]
        print(f"Significant positions: {len(significant)}")
        print(significant.sort_values(key=abs, ascending=False).head(5))
        return self.result


def main(use_excel_spec: bool,
         window: int,
         tracking_error: float,
         date1: str, 
         date2: str, 
         individual_limits: Optional[Dict[int, float]] = None, 
         exclude_stocks: Optional[List[int]] = None):
    runner = Runner(data_dir = "W_EMP008/DATA",
                    use_excel_spec = use_excel_spec,
                    window = window,
                    tracking_error = tracking_error)

    result = runner.run(
        date1=date1,
        date2=date2,
        individual_limits=individual_limits,
        exclude_stocks=exclude_stocks
    )
    return result, runner


if __name__ == "__main__":
    res, runner = main(use_excel_spec = True,
                       window = 36,
                       tracking_error = 0.7/np.sqrt(12),
                       date1="20250331", 
                       date2="20250430")

    # NOTE: For reporting purpose. Not used in the current version.
    # report_gen = ReportGenerator(res, runner)
    # report_filename = report_gen.generate_pdf_report()
    # print(f"Report saved: {report_filename}")

    print(f"\n=== Optimization Complete ===")
    print(f"Success: {res['success']}")
    print(f"Tracking Error: {res['tracking_error']:.6f}")
    print(f"Optimal weights: {res['optimal_weights']}")

    print("Used input for optimization:")
    print(runner.current_factors)
    print(runner.current_covariance)
    print(runner.current_residual_cov)
    print(runner.current_alpha)

