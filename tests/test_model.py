import torch

from veg_forecast.model import VegForecaster, quantile_loss


def _batch(B=3, Lh=12, Lf=40, F=30, last=29):
    torch.manual_seed(0)
    return {
        "history": torch.randn(B, Lh, F),
        "history_mask": torch.zeros(B, Lh, F, dtype=torch.bool),
        "history_pad_mask": torch.zeros(B, Lh, dtype=torch.bool),
        "future": torch.randn(B, Lf, F),
        "future_mask": torch.zeros(B, Lf, F, dtype=torch.bool),
        "future_pad_mask": torch.arange(Lf)[None].expand(B, -1) > last,
        "future_target_positions": torch.tensor([[4, 14, last]] * B),
    }


def _model():
    return VegForecaster(input_dim=30, quantiles=[0.1, 0.5, 0.9], num_targets=3, d_model=32, num_layers=2,
                         num_heads=4, dim_feedforward=64, dropout=0.0).eval()


def test_output_shape():
    with torch.no_grad():
        out = _model()(_batch())
    assert out.shape == (3, 3, 3, 3) and torch.isfinite(out).all()


def test_future_encoder_is_causal():
    """A forecast depends on the weather up to its own day only, and not on where the window ends."""
    m, b = _model(), _batch()
    with torch.no_grad():
        p0 = m(b)
        later = dict(b, future=b["future"].clone())
        later["future"][:, 5:] += 3.0  # weather after the first target day (position 4)
        p1 = m(later)
        longer = dict(b, future_pad_mask=torch.arange(40)[None].expand(3, -1) > 35)
        p2 = m(longer)
    assert torch.allclose(p1[:, 0], p0[:, 0], atol=1e-5)  # first target unchanged
    assert not torch.allclose(p1[:, 1], p0[:, 1], atol=1e-4)  # later targets do see it
    assert torch.allclose(p2, p0, atol=1e-5)


def test_history_padding_is_ignored():
    m, b = _model(), _batch()
    padded = dict(b)
    padded["history"] = torch.cat([b["history"], torch.randn(3, 4, 30)], dim=1)
    padded["history_mask"] = torch.cat([b["history_mask"], torch.ones(3, 4, 30, dtype=torch.bool)], dim=1)
    padded["history_pad_mask"] = torch.cat([b["history_pad_mask"], torch.ones(3, 4, dtype=torch.bool)], dim=1)
    with torch.no_grad():
        assert torch.allclose(m(b), m(padded), atol=1e-5)


def test_quantile_loss():
    preds = torch.tensor([[[0.0, 1.0, 2.0]]])  # (B=1, S=1, Q=3)
    target = torch.tensor([[1.5]])
    loss = quantile_loss(preds, target, torch.zeros(1, 1, dtype=torch.bool), [0.1, 0.5, 0.9])
    expected = (0.1 * 1.5 + 0.5 * 0.5 + 0.1 * 0.5) / 3
    assert abs(loss.item() - expected) < 1e-6
    masked = quantile_loss(preds, target, torch.ones(1, 1, dtype=torch.bool), [0.1, 0.5, 0.9])
    assert masked.item() == 0.0
