from unittest.mock import MagicMock

import pandas as pd
import torch

from data_pipeline.realtime_provider import RealtimeDataProvider


def test_build_feat_tensor_supports_registered_derived_feature(monkeypatch):
    monkeypatch.setattr(
        "model_core.config.ModelConfig.INPUT_FEATURES",
        ["LOG_MONEYNESS"],
        raising=False,
    )

    provider = RealtimeDataProvider(sql_engine=MagicMock())
    realtime_quotes = pd.DataFrame()
    cb_features = pd.DataFrame(
        {
            "code": ["123001.SZ", "127050.SZ"],
            "trade_date": ["2026-03-07", "2026-03-07"],
            "close_stk": [10.0, 20.0],
            "conv_price": [8.0, 16.0],
        }
    )

    feat_tensor = provider.build_feat_tensor(realtime_quotes, cb_features)
    assert feat_tensor.shape == (2, 1)
    assert torch.isfinite(feat_tensor).all()
    assert float(feat_tensor.abs().sum().item()) > 0.0


def test_build_feat_tensor_with_history_supports_registered_derived_feature(monkeypatch):
    monkeypatch.setattr(
        "model_core.config.ModelConfig.INPUT_FEATURES",
        ["LOG_MONEYNESS"],
        raising=False,
    )

    provider = RealtimeDataProvider(sql_engine=MagicMock())
    monkeypatch.setattr(
        provider,
        "get_trading_days_before",
        lambda date, window: ["2026-03-06", "2026-03-07"],
    )
    monkeypatch.setattr(
        provider,
        "get_cb_features_multi_days",
        lambda trading_days: pd.DataFrame(
            {
                "trade_date": ["2026-03-06", "2026-03-06", "2026-03-07", "2026-03-07"],
                "code": ["123001.SZ", "127050.SZ", "123001.SZ", "127050.SZ"],
                "close_stk": [10.0, 20.0, 11.0, 22.0],
                "conv_price": [8.0, 16.0, 8.0, 16.0],
            }
        ),
    )

    feat_tensor, asset_list = provider.build_feat_tensor_with_history(
        date="2026-03-07",
        realtime_quotes=pd.DataFrame(),
        window=2,
    )

    assert asset_list == ["123001.SZ", "127050.SZ"]
    assert feat_tensor.shape == (2, 2, 1)
    assert torch.isfinite(feat_tensor).all()
    assert float(feat_tensor.abs().sum().item()) > 0.0
