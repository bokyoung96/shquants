import numpy as np
import pandas as pd
from W_EMP008.regression import MultiFactorRegression


class CovarianceMatrix(MultiFactorRegression):
    def __init__(self, data_dir: str, use_excel_spec: bool = True) -> None:
        super().__init__(data_dir, use_excel_spec=use_excel_spec)

    def _get_regression_tickers(self) -> list[str]:
        coef = self._get_coefficients()
        return [c for c in coef.columns if c != 'date']

    def _get_regression_dates(self) -> list[pd.Timestamp]:
        coef = self._get_coefficients()
        return [pd.to_datetime(idx) for idx in coef.index]

    def _get_coefficients(self) -> pd.DataFrame:
        coef, _ = self.run_regression()
        return coef

    def get_matrix(self, date: str | pd.Timestamp, periods: int = 36) -> pd.DataFrame:
        coef = self._get_coefficients()
        tickers = self._get_regression_tickers()
        dates = self._get_regression_dates()
        if isinstance(date, str):
            date = pd.to_datetime(date)
        if date not in dates:
            raise ValueError(f"date {date} not in regression dates")
        
        end_idx = dates.index(date) + 1
        start_idx = max(0, end_idx - periods)
        use_dates = dates[start_idx:end_idx]
        
        coef_slice = coef.loc[[d.strftime('%Y%m%d') for d in use_dates], tickers]
        
        arr = coef_slice.to_numpy(dtype=np.float64)
        mask = np.isfinite(arr)
        arr = np.where(mask, arr, np.nan)
        cov = coef_slice.astype(float).cov(ddof=0)

        ret = self._get_preprocessed_data("Ret")
        n = len([i for i in ret.index if i not in ("IKS270", "IKS200")])
        factor = n / (n - 1) if n > 1 else 1.0
        cov *= factor
        return pd.DataFrame(cov, index=tickers, columns=tickers)

    @property
    def tickers(self) -> list[str]:
        return self._get_regression_tickers()

    @property
    def regression_dates(self) -> list[pd.Timestamp]:
        return self._get_regression_dates()


if __name__ == "__main__":
    print("--- Testing CovarianceMatrix with dynamic pivot ---")
    cm_dynamic = CovarianceMatrix("W_EMP008/DATA", use_excel_spec=False)
    try:
        last_date = cm_dynamic.regression_dates[-2]
        cov = cm_dynamic.get_matrix(last_date, periods=36)
        print(f"Covariance shape: {cov.shape}")
        print(f"Tickers: {cov.index[:5].tolist()} ...")
        print(cov.iloc[:5, :5])
    except Exception as e:
        print(f"Error: {e}")
    
    print("-" * 50)
    
    if cm_dynamic.zero_spec_from_file:
        print("--- Testing CovarianceMatrix with excel spec ---")
        cm_spec = CovarianceMatrix("W_EMP008/DATA", use_excel_spec=True)
        try:
            last_date = cm_spec.regression_dates[-2]
            cov = cm_spec.get_matrix(last_date, periods=36)
            print(f"Covariance shape: {cov.shape}")
            print(f"Tickers: {cov.index[:5].tolist()} ...")
            print(cov.iloc[:5, :5])
        except Exception as e:
            print(f"Error: {e}")
    else:
        print("--- Skipping spec test: spec file not found ---")
