import pandas as pd
import numpy as np
from tqdm import tqdm
from W_EMP008.regression import MultiFactorRegression


class Residuals(MultiFactorRegression):
    def __init__(self, data_dir: str, use_excel_spec: bool = True) -> None:
        super().__init__(data_dir, use_excel_spec=use_excel_spec)
        self.residuals_df = None
    
    def get_residuals(self) -> pd.DataFrame:
        coefficients_df, _ = self.run_regression()
        excess_returns = self.excess_returns
        
        factor_dates = list(self.pp_factors.keys())
        ret_dates = list(excess_returns.columns)
        
        residuals_list = []
        
        for i, (f_date, r_date) in enumerate(zip(factor_dates, ret_dates)):
            if f_date not in coefficients_df.index:
                continue
                
            factor_data = self.pp_factors[f_date]
            coefficients = coefficients_df.loc[f_date]
            returns = excess_returns[r_date]
            
            common_tickers = factor_data.index.intersection(returns.index)
            
            residuals_dict = {'date': r_date}
            
            for ticker in tqdm(common_tickers, desc="Calculating residuals per ticker"):
                factor_values = factor_data.loc[ticker]
                factor_values = factor_values.dropna()
                
                common_factors = factor_values.index.intersection(coefficients.index)
                if len(common_factors) == 0:
                    residuals_dict[ticker] = returns[ticker]
                    continue
                
                predicted_return = (factor_values[common_factors] * coefficients[common_factors]).sum()
                actual_return = returns[ticker]
                residual = actual_return - predicted_return
                
                residuals_dict[ticker] = residual
            
            residuals_list.append(residuals_dict)
        
        self.residuals_df = pd.DataFrame(residuals_list).set_index('date')
        return self.residuals_df
    
    def get_residual_covariance_matrix(self, window: int = 36, end_date: str = None) -> pd.DataFrame:
        # NOTE: Assumes residuals are centered at 0, i.e. mean = 0.
        if self.residuals_df is None:
            self.get_residuals()
        
        if end_date is None:
            recent_residuals = self.residuals_df.tail(window)
        else:
            try:
                end_idx = self.residuals_df.index.get_loc(end_date)
                start_idx = max(0, end_idx - window + 1)
                recent_residuals = self.residuals_df.iloc[start_idx:end_idx + 1]
            except KeyError:
                raise ValueError(f"Date {end_date} not found in residuals data")
        
        variances = (recent_residuals ** 2).sum() / len(recent_residuals)
        
        n_stocks = len(variances)
        cov_matrix = np.zeros((n_stocks, n_stocks))
        np.fill_diagonal(cov_matrix, variances.values)
        
        cov_df = pd.DataFrame(
            cov_matrix,
            index=variances.index,
            columns=variances.index
        )     
        return cov_df

    @property
    def residuals(self) -> pd.DataFrame:
        if self.residuals_df is None:
            self.get_residuals()
        return self.residuals_df

    @property
    def residuals_covariance(self) -> pd.DataFrame:
        return self.get_residual_covariance_matrix()


if __name__ == "__main__":    
    print("--- Residuals with LINEST-like dynamic pivot ---")
    residuals_dynamic = Residuals("W_EMP008/DATA", use_excel_spec=False)
    residuals_df_dynamic = residuals_dynamic.get_residuals()
    print("Residuals shape:", residuals_df_dynamic.shape)
    print(f"Sample residuals for first date:\n{residuals_df_dynamic.iloc[0].head()}")
    print("-" * 50)
    
    if residuals_dynamic.zero_spec_from_file:
        print("--- Residuals with specification from excel ---")
        residuals_spec = Residuals("W_EMP008/DATA", use_excel_spec=True)
        residuals_df_spec = residuals_spec.get_residuals()
        print("Residuals shape:", residuals_df_spec.shape)
        print(f"Sample residuals for first date:\n{residuals_df_spec.iloc[0].head()}")
    else:
        print("--- Skipping residuals with spec file: spec file not found. ---")

    print("--- Residuals covariance matrix ---")
    residuals_covariance = residuals_dynamic.get_residual_covariance_matrix(window=36,
                                                                            end_date="20250331")
    print(residuals_covariance)
