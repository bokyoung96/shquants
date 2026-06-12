import numpy as np
import pandas as pd
from W_EMP008.loader import DataLoader


class FactorPreProcess:
    _preprocessing_rules = {
        "FlMktcap": {"drop_indices": ["IKS200"]},
        "Ret": {"drop_indices": ["IKS200"]},
        "Sector": {"drop_indices": []},
        "LnMktcap": {
            "drop_indices": ["IKS200"], 
            "log_transform": True, 
            "rank_transform": True
        },
        "BM": {"exclude": True},
        "default": {"drop_indices": ["IKS200"]}
    }
    
    def __init__(self, data_dir: str) -> None:
        self.loader = DataLoader(data_dir)
        self.fl = self._get_preprocessed_data("FlMktcap")
        self.factors = [
            f for f in self.loader.available()
            if not self._preprocessing_rules.get(f, {}).get("exclude", False)
            and f not in ("Ret", "FlMktcap", "Sector")
        ]
        self.fl_weight = self.fl.div(self.fl.sum(axis=0), axis=1)
        self.factor_weighted_means, self.factor_filled = self._get_factors()

    def _get_preprocessed_data(self, name: str) -> pd.DataFrame:
        df = self.loader.__getattr__(name)
        rules = self._preprocessing_rules.get(name, self._preprocessing_rules["default"])
        
        if rules.get("drop_indices"):
            df = df.drop(index=rules["drop_indices"], errors="ignore")
        return df

    def _apply_transforms(self, df: pd.DataFrame, name: str) -> pd.DataFrame:
        rules = self._preprocessing_rules.get(name, {})
        
        if rules.get("log_transform") or name.startswith("Ln"):
            df = np.log(df)
        return df

    def _apply_post_fill_transforms(self, filled: pd.DataFrame, name: str) -> tuple[pd.DataFrame, pd.Series]:
        rules = self._preprocessing_rules.get(name, {})
        
        if rules.get("rank_transform"):
            filled = filled.rank(axis=0, method="min", ascending=True)
            mean_vals = filled.apply(lambda col: (col * self.fl_weight[col.name]).sum() / self.fl_weight[col.name].sum() if self.fl_weight[col.name].sum() != 0 else np.nan)
            return filled, mean_vals
        return filled, None

    def _get_factors(self) -> tuple[dict[str, pd.Series], dict[str, pd.DataFrame]]:
        factor_filled = {}
        factor_weighted_means = {}
        
        for name in self.factors:
            df = self._get_preprocessed_data(name)
            df = self._apply_transforms(df, name)
            
            filled = df.copy()
            mean_vals = pd.Series(index=filled.columns, dtype=float)

            for col in filled.columns:
                notna = filled[col].notna()
                w = self.fl_weight.loc[notna, col]
                v = filled.loc[notna, col]
                if w.sum() == 0:
                    mean_val = np.nan
                else:
                    mean_val = (v * w).sum() / w.sum()
                mask = filled[col].isna()
                filled.loc[mask, col] = mean_val
                mean_vals[col] = mean_val

            filled, post_mean_vals = self._apply_post_fill_transforms(filled, name)
            if post_mean_vals is not None:
                mean_vals = post_mean_vals

            factor_filled[name] = filled
            factor_weighted_means[name] = mean_vals
        return factor_weighted_means, factor_filled

    def _get_zscores(self) -> dict[str, pd.DataFrame]:
        zscores = {}
        for name, df in self.factor_filled.items():            
            base = self.factor_weighted_means[name]
            std = ((df.sub(base, axis=1) ** 2).sum(axis=0) / (df.shape[0])).pow(0.5)
            z = df.sub(base, axis=1).div(std, axis=1)
            zscores[name] = z

        ret = self._get_preprocessed_data("Ret")
        ret_na_mask = ret.isna()
        factor_cols = next(iter(self.factor_filled.values())).columns
        
        if len(ret.columns) > len(factor_cols):
            ret_na_mask_aligned = ret_na_mask[ret_na_mask.columns[1:len(factor_cols)+1]]
        else:
            ret_na_mask_aligned = ret_na_mask
        
        ret_na_mask_aligned.columns = factor_cols
        
        for name in zscores:
            zscores[name][ret_na_mask_aligned] = 0
        return zscores

    def _get_sectors(self) -> pd.DataFrame:
        sector_df = self._get_preprocessed_data("Sector")
        sector_names = sector_df.iloc[:, 0]
        fl_weight_with_sector = self.fl_weight.copy()
        fl_weight_with_sector.index = pd.MultiIndex.from_arrays([
            fl_weight_with_sector.index,
            sector_names.reindex(fl_weight_with_sector.index).fillna("").values
        ], names=["ticker", "sector"])
        sector_weight_sum = fl_weight_with_sector.groupby("sector").sum().fillna(0)
        return sector_weight_sum

    def _get_sector_dummies(self) -> dict[str, pd.DataFrame]:
        sector_df = self._get_preprocessed_data("Sector")
        sector_names = sector_df.iloc[:, 0]
        
        factor_tickers = next(iter(self.factor_filled.values())).index
        factor_dates = next(iter(self.factor_filled.values())).columns
        unique_sectors = sector_names.unique()
        
        sector_dummies = {}
        
        for sector in unique_sectors:
            dummy_df = pd.DataFrame(0, index=factor_tickers, columns=factor_dates)
            
            sector_tickers = sector_names[sector_names == sector].index
            matching_tickers = dummy_df.index.intersection(sector_tickers)
            dummy_df.loc[matching_tickers, :] = 1
            
            sector_dummies[sector] = dummy_df
        
        return sector_dummies

    def _get_sector_factors(self) -> dict[str, pd.DataFrame]:
        sector_dummies = self._get_sector_dummies()
        sector_weights = self._get_sectors()
        lnmktcap_zero_mask = self.zscores['LnMktcap'] == 0
        
        sector_factors = {}
        
        for sector in sector_dummies.keys():
            dummy_df = sector_dummies[sector]
            if sector in sector_weights.index:
                weight_series = sector_weights.loc[sector]
                sector_factor = dummy_df.sub(weight_series, axis=1)
            else:
                sector_factor = dummy_df
            sector_factor[lnmktcap_zero_mask] = 0
            sector_factors[sector] = sector_factor
        
        return sector_factors

    def _get_combined_factors(self) -> dict[str, pd.DataFrame]:
        zscores_data = self.zscores
        sector_data = self.sector_factors
        
        combined_factors = {}
        factor_dates = next(iter(zscores_data.values())).columns
        
        for date in factor_dates:
            date_data = []
            
            for factor_name, factor_df in zscores_data.items():
                if factor_name != "Ret":
                    date_data.append(factor_df[date].rename(factor_name))
            
            for sector_name, sector_df in sector_data.items():
                date_data.append(sector_df[date].rename(sector_name))
            
            combined_df = pd.concat(date_data, axis=1)
            combined_factors[date] = combined_df
        return combined_factors

    @property
    def zscores(self) -> dict[str, pd.DataFrame]:
        return self._get_zscores()

    @property
    def sectors(self) -> pd.DataFrame:
        return self._get_sectors()

    @property
    def sector_factors(self) -> dict[str, pd.DataFrame]:
        return self._get_sector_factors()

    @property
    def pp_factors(self) -> 'PPFactors':
        return PPFactors(self._get_combined_factors())


class PPFactors(dict):
    def __getitem__(self, date_str):
        if isinstance(date_str, str):
            target_date = pd.to_datetime(date_str)
            for date_key in self.keys():
                if pd.to_datetime(date_key) == target_date:
                    return super().__getitem__(date_key)
            available_dates = [pd.to_datetime(k).strftime('%Y%m%d') for k in list(self.keys())[:5]]
            raise KeyError(f"Date '{date_str}' not found. Available dates (first 5): {available_dates}")
        return super().__getitem__(date_str)


if __name__ == "__main__":
    pp = FactorPreProcess("W_EMP008/DATA")
    # df = pp.pp_factors['20250331']