import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Optional, List, Dict, Any
from W_EMP008.components import RiskModelComponents


class PortfolioOptimizer(RiskModelComponents):
    def __init__(self, 
                 data_dir: str,
                 use_excel_spec: bool = True,
                 window: int = 36,
                 tracking_error: float = 0.7/np.sqrt(12)):
        super().__init__(data_dir, use_excel_spec, window)
        self.tracking_error = tracking_error
        self.iteration_count = 0
        self._current_objective = 0.0
        
    def _progress_callback(self, x: np.ndarray) -> None:
        self.iteration_count += 1
        if self.iteration_count % 10 == 0:
            print(f"Iteration {self.iteration_count}: current objective = {self._current_objective:.6f}")
        
    def _build_covariance_matrix(self, 
                                factor_cov: pd.DataFrame,
                                factors: pd.DataFrame, 
                                residual_cov: pd.DataFrame) -> np.ndarray:
        z = factors.values
        cov = factor_cov.values
        D = residual_cov.values
        return D + z @ cov @ z.T
    
    def _objective_function(self, x: np.ndarray, alpha_exp: np.ndarray, factors: np.ndarray) -> float:
        obj_val = -(alpha_exp.T @ factors.T @ x)
        self._current_objective = obj_val
        return obj_val
    
    def _tracking_error_constraint(self, x: np.ndarray, M: np.ndarray, te_limit: float) -> float:
        variance = x.T @ M @ x
        return te_limit**2 - variance
    
    def _build_sector_constraints(self, factors: pd.DataFrame, n_sectors: int = 26) -> np.ndarray:
        z = factors.values
        z_sec = np.zeros((n_sectors, z.shape[0]))
        
        sector_start_idx = 3
        for i in range(n_sectors):
            factor_idx = sector_start_idx + i
            if factor_idx < z.shape[1]:
                z_sec[i, :] = z[:, factor_idx]
            else:
                break
        return z_sec
    
    def optimize_portfolio(self,
                          date1: str,
                          date2: str,
                          individual_limits: Optional[Dict[int, float]] = None,
                          exclude_stocks: Optional[List[int]] = None) -> Dict[str, Any]:        
        self.iteration_count = 0
        self._current_objective = 0.0
        
        factors = self.factors(date2)
        factor_cov = self.factor_covariance(date1, self.window)
        residual_cov = self.residual_covariance(end_date=date2)
        alpha_exp = self.expected_alpha(end_date=date1, window=self.window)
        index_wgt = self.bm.iloc[:, -1]
        
        common_tickers = factors.index.intersection(residual_cov.index).intersection(index_wgt.index)
        common_factors = factors.columns.intersection(factor_cov.index).intersection(alpha_exp.index)
        
        factors_aligned = factors.loc[common_tickers, common_factors]
        factor_cov_aligned = factor_cov.loc[common_factors, common_factors]
        residual_cov_aligned = residual_cov.loc[common_tickers, common_tickers]
        alpha_exp_aligned = alpha_exp[common_factors]
        index_wgt_aligned = index_wgt[common_tickers]
        
        M = self._build_covariance_matrix(factor_cov_aligned, factors_aligned, residual_cov_aligned)
        
        m = len(common_tickers)
        n_exclude = len(exclude_stocks) if exclude_stocks else 0
        n_individual = len(individual_limits) if individual_limits else 0
        n_constraints = 26 + 1 + n_individual + n_exclude
        
        z_sec = self._build_sector_constraints(factors_aligned)
        alpha_array = alpha_exp_aligned.values
        z_array = factors_aligned.values
        index_wgt_array = index_wgt_aligned.values
        
        Aeq = np.zeros((n_constraints, m))
        beq = np.zeros(n_constraints)
        
        # Long-Short constraint: sum(x) = 0
        Aeq[0, :] = 1.0
        beq[0] = 0.0
        
        constraint_idx = 1
        
        # Individual limits constraints: Equal weight as index
        if individual_limits:
            for stock_idx, limit in individual_limits.items():
                if stock_idx < m:
                    Aeq[constraint_idx, stock_idx] = 1.0
                    beq[constraint_idx] = limit
                    constraint_idx += 1
        
        # Exclude stocks constraints: x_i = -w_i (complete exclusion)
        if exclude_stocks:
            for stock_idx in exclude_stocks:
                if stock_idx < m:
                    Aeq[constraint_idx, stock_idx] = 1.0
                    beq[constraint_idx] = -index_wgt_array[stock_idx]
                    constraint_idx += 1
        
        # Sector neutrality constraints: sum(z_secᵢ @ x) = 0
        for i in range(26):
            Aeq[constraint_idx + i, :] = z_sec[i, :]
            beq[constraint_idx + i] = 0.0
        
        constraints = [
            {'type': 'eq', 'fun': lambda x, A=Aeq, b=beq: A @ x - b},
            {'type': 'ineq', 'fun': lambda x, M=M, te=self.tracking_error: self._tracking_error_constraint(x, M, te)}
        ]
        
        bounds = [(-index_wgt_array[i], None) for i in range(m)]
        
        x0 = np.zeros(m)
        
        print("Starting optimization...")
        result = minimize(
            fun=lambda x: self._objective_function(x, alpha_array, z_array),
            x0=x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            callback=self._progress_callback,
            options={
                'maxiter': int(1e8),
                'ftol': 1e-15,
                'eps': 1e-15,
                'disp': True,
                'finite_diff_rel_step': 1e-12
            }
        )
        
        print(f"Optimization completed after {self.iteration_count} iterations")
        
        optimal_weights = index_wgt_array + result.x
        
        return {
            'success': result.success,
            'optimal_weights': pd.Series(optimal_weights, index=common_tickers),
            'active_weights': pd.Series(result.x, index=common_tickers),
            'objective_value': -result.fun,
            'tracking_error': np.sqrt(result.x.T @ M @ result.x),
            'tickers': common_tickers.tolist(),
            'optimization_result': result
        }




if __name__ == "__main__":
    optimizer = PortfolioOptimizer("W_EMP008/DATA", 
                                   use_excel_spec=True, 
                                   window=36)
    
    result = optimizer.optimize_portfolio(
        date1="20250228",
        date2="20250331",
        individual_limits=None
    )