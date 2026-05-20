from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .tft_dataset import make_prediction_sequences, make_supervised_sequences


@dataclass(frozen=True, slots=True)
class TFTConfig:
    lookback: int = 252
    hidden_size: int = 16
    num_heads: int = 2
    num_layers: int = 1
    dropout: float = 0.25
    learning_rate: float = 1e-3
    weight_decay: float = 1e-3
    max_epochs: int = 60
    batch_size: int = 64
    validation_window: int = 252
    min_train_sequences: int = 256
    early_stopping_patience: int = 8
    min_delta: float = 1e-5
    seed: int = 42
    device: str = "cpu"


class _TemporalTransformer(nn.Module):
    def __init__(self, input_size: int, config: TFTConfig) -> None:
        super().__init__()
        self.input_projection = nn.Linear(input_size, config.hidden_size)
        self.position = nn.Parameter(torch.zeros(1, config.lookback, config.hidden_size))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_size * 4,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        self.norm = nn.LayerNorm(config.hidden_size)
        self.head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, 1),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        hidden = self.input_projection(values) + self.position[:, : values.shape[1], :]
        encoded = self.encoder(hidden)
        return self.head(self.norm(encoded[:, -1, :])).squeeze(-1)


@dataclass(slots=True)
class TemporalFusionTransformerForecaster:
    config: TFTConfig = TFTConfig()
    columns_: list[str] = field(init=False)
    fill_: pd.Series = field(init=False)
    mean_: pd.Series = field(init=False)
    std_: pd.Series = field(init=False)
    target_mean_: float = field(init=False)
    target_std_: float = field(init=False)
    target_min_: float = field(init=False)
    target_max_: float = field(init=False)
    model_: _TemporalTransformer = field(init=False)
    training_rows_: int = field(init=False)
    validation_rows_: int = field(init=False)
    best_iteration_: int = field(init=False)
    best_validation_loss_: float = field(init=False)

    def fit(self, features: pd.DataFrame, target: pd.Series) -> "TemporalFusionTransformerForecaster":
        if self.config.lookback <= 0:
            raise ValueError("lookback must be positive")
        if self.config.hidden_size % self.config.num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")

        torch.manual_seed(self.config.seed)
        torch.set_num_threads(1)

        aligned = features.join(target.rename("__target__"), how="inner").dropna(subset=["__target__"])
        if aligned.empty:
            raise ValueError("cannot fit temporal transformer without target observations")

        x = aligned.drop(columns=["__target__"]).astype(float)
        y = aligned["__target__"].astype(float)
        self.columns_ = list(x.columns)
        self.fill_ = x.median().fillna(0.0)
        x = x.fillna(self.fill_)
        self.mean_ = x.mean()
        self.std_ = x.std(ddof=0).replace(0.0, 1.0)
        x = (x - self.mean_) / self.std_
        self.target_mean_ = float(y.mean())
        target_std = float(y.std(ddof=0))
        self.target_std_ = target_std if abs(target_std) > 1e-12 else 1.0
        self.target_min_ = float(y.min())
        self.target_max_ = float(y.max())
        scaled_target = (y - self.target_mean_) / self.target_std_

        sequences = make_supervised_sequences(x, scaled_target, self.config.lookback)
        if len(sequences.targets) == 0:
            raise ValueError("not enough observations to build transformer sequences")

        train_x, train_y, valid_x, valid_y = self._split_validation_tail(sequences.values, sequences.targets)
        self.training_rows_ = len(train_y)
        self.validation_rows_ = len(valid_y)
        if self.training_rows_ == 0:
            raise ValueError("not enough train sequences after validation split")

        device = torch.device(self.config.device)
        self.model_ = _TemporalTransformer(input_size=len(self.columns_), config=self.config).to(device)
        optimizer = torch.optim.AdamW(
            self.model_.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        loss_fn = nn.MSELoss()
        loader = DataLoader(
            TensorDataset(torch.as_tensor(train_x), torch.as_tensor(train_y)),
            batch_size=self.config.batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(self.config.seed),
        )
        valid_tensor_x = torch.as_tensor(valid_x, device=device) if len(valid_y) else None
        valid_tensor_y = torch.as_tensor(valid_y, device=device) if len(valid_y) else None

        best_state = {key: value.detach().cpu().clone() for key, value in self.model_.state_dict().items()}
        best_loss = float("inf")
        best_epoch = 0
        stale_epochs = 0
        for epoch in range(1, self.config.max_epochs + 1):
            self.model_.train()
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = loss_fn(self.model_(batch_x), batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model_.parameters(), max_norm=1.0)
                optimizer.step()

            current_loss = self._validation_loss(valid_tensor_x, valid_tensor_y, train_x, train_y, loss_fn, device)
            if current_loss < best_loss - self.config.min_delta:
                best_loss = current_loss
                best_epoch = epoch
                stale_epochs = 0
                best_state = {key: value.detach().cpu().clone() for key, value in self.model_.state_dict().items()}
            else:
                stale_epochs += 1
                if len(valid_y) and stale_epochs >= self.config.early_stopping_patience:
                    break

        self.model_.load_state_dict(best_state)
        self.best_iteration_ = best_epoch
        self.best_validation_loss_ = float(best_loss)
        return self

    def predict(self, features: pd.DataFrame) -> pd.Series:
        return self.predict_with_context(features, features.index)

    def predict_with_context(self, context_features: pd.DataFrame, prediction_index: pd.Index) -> pd.Series:
        missing = [column for column in self.columns_ if column not in context_features.columns]
        if missing:
            raise ValueError(f"missing feature columns for prediction: {', '.join(missing)}")

        x = context_features.loc[:, self.columns_].astype(float).fillna(self.fill_)
        x = (x - self.mean_) / self.std_
        sequences, dates = make_prediction_sequences(x, prediction_index, self.config.lookback, self.columns_)
        if len(dates) == 0:
            return pd.Series(dtype=float, name="prediction")

        device = torch.device(self.config.device)
        self.model_.eval()
        predictions = []
        with torch.no_grad():
            for start in range(0, len(sequences), self.config.batch_size):
                batch = torch.as_tensor(sequences[start : start + self.config.batch_size], device=device)
                predictions.append(self.model_(batch).detach().cpu().numpy())
        values = np.concatenate(predictions) * self.target_std_ + self.target_mean_
        values = np.clip(values, self.target_min_, self.target_max_)
        return pd.Series(values, index=dates, name="prediction")

    def _split_validation_tail(
        self, values: np.ndarray, targets: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        validation_window = int(self.config.validation_window)
        min_train = int(self.config.min_train_sequences)
        if validation_window <= 0 or len(targets) < min_train + validation_window:
            return values, targets, values[:0], targets[:0]
        train_end = len(targets) - validation_window
        return values[:train_end], targets[:train_end], values[train_end:], targets[train_end:]

    def _validation_loss(
        self,
        valid_x: torch.Tensor | None,
        valid_y: torch.Tensor | None,
        train_x: np.ndarray,
        train_y: np.ndarray,
        loss_fn: nn.Module,
        device: torch.device,
    ) -> float:
        self.model_.eval()
        with torch.no_grad():
            if valid_x is not None and valid_y is not None and len(valid_y):
                return float(loss_fn(self.model_(valid_x), valid_y).detach().cpu())
            train_tensor_x = torch.as_tensor(train_x, device=device)
            train_tensor_y = torch.as_tensor(train_y, device=device)
            return float(loss_fn(self.model_(train_tensor_x), train_tensor_y).detach().cpu())
