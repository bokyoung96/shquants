import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
from W_EMP008.preprocess import FactorPreProcess


class MultiFactorRegression(FactorPreProcess):
    def __init__(self, data_dir: str, use_excel_spec: bool = True) -> None:
        super().__init__(data_dir)
        self.use_excel_spec = use_excel_spec
        self.coefficients_df = None
        self.r2_df = None
        self.zero_spec_from_file = self._load_zero_spec(data_dir)
        
    def _load_zero_spec(self, data_dir: str) -> dict[str, str] | None:
        spec_path = os.path.join(data_dir, "zero_spec.xlsx")
        if not os.path.exists(spec_path):
            return None
        try:
            df = pd.read_excel(spec_path)
            df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0]).dt.strftime('%Y%m%d')
            return pd.Series(df.iloc[:, 1].values, index=df.iloc[:, 0]).to_dict()
        except Exception:
            return None
        
    def _get_linest_independent_cols(self, df: pd.DataFrame) -> list[int]:
        independent_cols_indices = []
        matrix = df.to_numpy()
        current_rank = 0
        for i in range(matrix.shape[1]):
            cols_to_test = independent_cols_indices + [i]
            tol = np.finfo(matrix.dtype).eps * max(matrix.shape)
            new_rank = np.linalg.matrix_rank(matrix[:, cols_to_test], tol=tol)
            if new_rank > current_rank:
                independent_cols_indices.append(i)
                current_rank = new_rank
        return independent_cols_indices

    def _get_excess_returns(self) -> pd.DataFrame:
        ret_data = self._get_preprocessed_data("Ret")
        
        if "IKS270" in ret_data.index:
            ret_data = ret_data.drop(index="IKS270")
        
        common_tickers = ret_data.index.intersection(self.fl_weight.index)
        ret_aligned = ret_data.loc[common_tickers].fillna(0)
        weight_aligned = self.fl_weight.loc[common_tickers]
        
        benchmark_returns = pd.Series(index=ret_aligned.columns, dtype=float)
        
        for i, date in enumerate(ret_aligned.columns):
            if i < len(weight_aligned.columns):
                benchmark_return = ret_aligned.iloc[:, i].mul(weight_aligned.iloc[:, i]).sum()
                benchmark_returns[date] = benchmark_return
            else:
                benchmark_returns[date] = 0
        
        excess_returns = ret_aligned.sub(benchmark_returns, axis=1)
        nan_mask = ret_data.isna() | benchmark_returns.isna()
        excess_returns[nan_mask] = 0
        return excess_returns.fillna(0)
    
    def run_regression(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        # NOTE: Multi-Collinearity considered by two methods.

        pp_factors_data = self.pp_factors
        excess_returns_data = self._get_excess_returns()
        
        factor_dates = list(pp_factors_data.keys())
        ret_dates = list(excess_returns_data.columns)
        
        coefficients_list = []
        r2_list = []
        
        for f_date, r_date in zip(factor_dates, ret_dates):
            X_data = pp_factors_data[f_date].dropna(how='all', axis=1)
            y_data = excess_returns_data[r_date]

            common_index = X_data.index.intersection(y_data.index)
            X_data = X_data.loc[common_index]
            y_data = y_data.loc[common_index]
            
            if len(X_data) == 0 or X_data.shape[1] == 0 or y_data.empty:
                continue

            run_with_spec = self.use_excel_spec and self.zero_spec_from_file and f_date in self.zero_spec_from_file
            
            if run_with_spec:
                factor_to_zero = self.zero_spec_from_file[f_date]
                if factor_to_zero in X_data.columns:
                    X_model = X_data.drop(columns=[factor_to_zero])
                    
                    model = sm.OLS(y_data, X_model)
                    result = model.fit(method='qr')
                    
                    all_coef = pd.Series(0.0, index=X_data.columns)
                    all_coef.update(result.params)
                else:
                    run_with_spec = False
            
            if not run_with_spec:
                independent_col_indices = self._get_linest_independent_cols(X_data)
                X_independent = X_data.iloc[:, independent_col_indices]
                
                model = sm.OLS(y_data, X_independent)
                result = model.fit(method='qr')
                
                all_coef = pd.Series(0.0, index=X_data.columns)
                all_coef.update(result.params)
            
            coef_dict = all_coef.to_dict()
            coef_dict['date'] = f_date
            coefficients_list.append(coef_dict)
            
            r2_dict = {'date': f_date, 'r2': result.rsquared}
            r2_list.append(r2_dict)
        
        self.coefficients_df = pd.DataFrame(coefficients_list).set_index('date')
        self.r2_df = pd.DataFrame(r2_list).set_index('date')
        
        return self.coefficients_df, self.r2_df
    
    @property
    def coefficients(self) -> pd.DataFrame:
        if self.coefficients_df is None:
            self.run_regression()
        return self.coefficients_df
    
    @property
    def r2_scores(self) -> pd.DataFrame:
        if self.r2_df is None:
            self.run_regression()
        return self.r2_df

    @property
    def excess_returns(self) -> pd.DataFrame:
        return self._get_excess_returns()


if __name__ == "__main__":
    print("--- Running regression with LINEST-like dynamic pivot ---")
    mfr_dynamic = MultiFactorRegression("W_EMP008/DATA", use_excel_spec=False)
    coef_df_dynamic, r2_df_dynamic = mfr_dynamic.run_regression()
    print("Coefficients shape:", coef_df_dynamic.shape)
    print("R2 scores shape:", r2_df_dynamic.shape)
    print("-" * 50)

    if mfr_dynamic.zero_spec_from_file:
        print("--- Running regression with specification from multi_col_excel_res.xlsx ---")
        mfr_spec = MultiFactorRegression("W_EMP008/DATA", use_excel_spec=True)
        coef_df_spec, r2_df_spec = mfr_spec.run_regression()
        print("Coefficients shape:", coef_df_spec.shape)
        print("R2 scores shape:", r2_df_spec.shape)
        
        sample_date = next(iter(mfr_spec.zero_spec_from_file))
        if sample_date in coef_df_spec.index:
            print(f"\nCoefficients for a sample date ({sample_date}) with specified zero:")
            print(coef_df_spec.loc[sample_date])
            print(f"Specified zero factor: {mfr_spec.zero_spec_from_file[sample_date]}")