"""Dual-branch Transformer for probabilistic multi-output vegetation-index forecasting.

History branch: Transformer encoder over the daily history window (weather, calendar and engineered features, plus
the VI values on clear-sky days; VIs are masked on the other days).

Future branch: Transformer encoder over the daily future weather with a CAUSAL mask, so the representation of day
d depends on weather up to day d only. Every future day is a query: it is enriched with an embedding of the days
elapsed since the last historical observation (dt) and retrieves context from the encoded history through
cross-attention. One quantile head per VI maps [query, context] to the requested quantiles.

The model therefore produces a forecast for every day of the future grid; training and evaluation select the days
with a valid clear-sky observation (`future_target_positions`), which are never an input of the model.
"""
import math

import torch
import torch.nn as nn

from .layers import PositionalEncoding

DT_MAX = 70.0  # longest future window of the training caches (days); scale of the dt features


class VegForecaster(nn.Module):
    """
    Args:
        input_dim: number of input features (identical for history and future tokens).
        quantiles: increasing quantile levels in (0, 1), e.g. [0.1, 0.5, 0.9].
        num_targets: number of vegetation indices predicted jointly.
        d_model, num_layers, num_heads, dim_feedforward, dropout: Transformer hyper-parameters (shared by the two
            encoders, the cross-attention and the heads).

    Input (dict, as built by `veg_forecast.data.collate_batch`):
        history (B, Th, F), history_mask (B, Th, F) True = missing, history_pad_mask (B, Th) True = padding,
        future (B, Tf, F), future_mask (B, Tf, F), future_pad_mask (B, Tf),
        future_target_positions (B, S) indices of the future days to forecast. Position p is p + 1 days after the
        last historical observation (the history window always ends with a valid observation).

    Output: (B, S, num_targets, Q) quantiles in the scaled target space.
    """

    def __init__(self, input_dim, quantiles, num_targets, d_model=128, num_layers=8, num_heads=8,
                 dim_feedforward=512, dropout=0.1):
        super().__init__()
        quantiles = [float(q) for q in quantiles]
        if not quantiles or sorted(set(quantiles)) != quantiles or not all(0.0 < q < 1.0 for q in quantiles):
            raise ValueError(f"quantiles must be distinct, increasing and in (0, 1): {quantiles}")
        if num_targets < 1:
            raise ValueError("num_targets must be >= 1")
        self.input_dim = input_dim
        self.quantiles = quantiles
        self.num_targets = num_targets
        self.d_model = d_model

        self.history_proj = nn.Linear(input_dim, d_model)
        self.future_proj = nn.Linear(input_dim, d_model)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=dim_feedforward,
                                           dropout=dropout, batch_first=True, norm_first=True, activation="relu")
        self.history_encoder = nn.TransformerEncoder(layer, num_layers=num_layers, enable_nested_tensor=False)
        self.future_encoder = nn.TransformerEncoder(layer, num_layers=num_layers, enable_nested_tensor=False)
        self.history_pos = PositionalEncoding(d_model=d_model, dropout=dropout)
        self.future_pos = PositionalEncoding(d_model=d_model, dropout=dropout)

        self.dt_proj = nn.Linear(2, d_model)
        self.history_xattn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.history_xnorm = nn.LayerNorm(d_model)

        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(2 * d_model, d_model),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, len(quantiles)),
            )
            for _ in range(num_targets)
        ])

    @staticmethod
    def _clean(sequence, mask):
        """Missing inputs enter the projection as 0."""
        return torch.where(mask, torch.zeros_like(sequence), sequence)

    @staticmethod
    def dt_features(positions):
        """[log1p(dt), dt] / scale, with dt = days since the last historical observation."""
        dt = (positions + 1).float()
        return torch.stack([torch.log1p(dt) / math.log1p(DT_MAX), dt / DT_MAX], dim=-1)

    def encode_history(self, history, history_mask, history_pad_mask):
        """Returns the encoded tokens (B, Th, d) and the key-padding mask (padding or fully missing tokens)."""
        ignore = history_pad_mask | history_mask.all(dim=-1)
        emb = self.history_pos(self.history_proj(self._clean(history, history_mask)))
        return self.history_encoder(emb, src_key_padding_mask=ignore), ignore

    def encode_future(self, future, future_mask, future_pad_mask):
        """Causal encoding of the daily future covariates (B, Tf, d)."""
        ignore = future_pad_mask | future_mask.all(dim=-1)
        length = future.size(1)
        causal = torch.triu(torch.ones(length, length, dtype=torch.bool, device=future.device), diagonal=1)
        emb = self.future_pos(self.future_proj(self._clean(future, future_mask)))
        return self.future_encoder(emb, mask=causal, src_key_padding_mask=ignore)

    def forward(self, batch):
        h_encoded, h_ignore = self.encode_history(batch["history"], batch["history_mask"], batch["history_pad_mask"])
        f_encoded = self.encode_future(batch["future"], batch["future_mask"], batch["future_pad_mask"])

        pos = batch["future_target_positions"]  # (B, S)
        selected = torch.gather(f_encoded, 1, pos.unsqueeze(-1).expand(-1, -1, self.d_model))
        query = selected + self.dt_proj(self.dt_features(pos))
        attended, _ = self.history_xattn(query, h_encoded, h_encoded, key_padding_mask=h_ignore)
        context = self.history_xnorm(attended)

        fusion = torch.cat([query, context], dim=-1)
        return torch.stack([head(fusion) for head in self.heads], dim=2)  # (B, S, K, Q)

    @classmethod
    def from_config(cls, config):
        keys = ("input_dim", "quantiles", "num_targets", "d_model", "num_layers", "num_heads", "dim_feedforward",
                "dropout")
        return cls(**{k: config[k] for k in keys})


def quantile_loss(preds, targets, mask, quantiles, weights=None):
    """Pinball loss of one VI, averaged over quantiles and (weighted) valid items.

    preds (B, S, Q), targets (B, S), mask (B, S) True = ignore, weights (B, S) optional non-negative weights.
    """
    if preds.size(-1) != len(quantiles):
        raise ValueError("last dimension of preds must match the number of quantiles")
    w = (~mask).float()
    if weights is not None:
        w = w * weights
    diff = targets.unsqueeze(-1) - preds  # (B, S, Q)
    tau = torch.as_tensor(quantiles, dtype=preds.dtype, device=preds.device)
    loss = torch.where(diff >= 0, tau * diff, (tau - 1) * diff)
    return (loss * w.unsqueeze(-1)).sum() / (w.sum().clamp(min=1e-8) * len(quantiles))


def multi_target_quantile_loss(preds, targets, mask, quantiles, weights=None):
    """Per-VI pinball losses (list of K scalars); the training objective is their sum.

    preds (B, S, K, Q), targets (B, S, K), mask (B, S, K), weights (B, S).
    """
    return [quantile_loss(preds[:, :, k], targets[:, :, k], mask[:, :, k], quantiles, weights)
            for k in range(preds.size(2))]
