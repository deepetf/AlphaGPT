import os
import sys

import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_core.data_loader import CBDataLoader


def _build_minimal_cb_df(include_future_bond: bool = False) -> pd.DataFrame:
    rows = [
        {
            "trade_date": "2026-01-02",
            "code": "110001.SH",
            "name": "转债A",
            "close": 100.0,
            "vol": 1000.0,
            "left_years": 1.2,
            "dblow": 10.0,
        },
        {
            "trade_date": "2026-01-03",
            "code": "110001.SH",
            "name": "转债A",
            "close": 101.0,
            "vol": 1000.0,
            "left_years": 1.2,
            "dblow": 11.0,
        },
        {
            "trade_date": "2026-01-02",
            "code": "110002.SH",
            "name": "转债B",
            "close": 110.0,
            "vol": 1500.0,
            "left_years": 1.1,
            "dblow": 20.0,
        },
        {
            "trade_date": "2026-01-03",
            "code": "110002.SH",
            "name": "转债B",
            "close": 111.0,
            "vol": 1500.0,
            "left_years": 1.1,
            "dblow": 19.0,
        },
    ]

    if include_future_bond:
        rows.append(
            {
                "trade_date": "2026-01-03",
                "code": "110003.SH",
                "name": "未来新债",
                "close": 120.0,
                "vol": 2000.0,
                "left_years": 1.0,
                "dblow": 5.0,
            }
        )

    return pd.DataFrame(rows)


def test_data_loader_builds_stage_a_masks(monkeypatch):
    monkeypatch.setattr(
        "model_core.config.ModelConfig.INPUT_FEATURES",
        ["DBLOW_CS_RANK"],
        raising=False,
    )
    monkeypatch.setattr(
        "model_core.config.ModelConfig.WARMUP_DAYS",
        0,
        raising=False,
    )
    monkeypatch.setattr(
        "model_core.data_loader.pd.read_parquet",
        lambda _: _build_minimal_cb_df(include_future_bond=True),
    )

    loader = CBDataLoader()
    loader.load_data(start_date="2026-01-02")

    assert loader.assets_list == ["110001.SH", "110002.SH", "110003.SH"]
    assert torch.equal(
        loader.listed_mask.cpu(),
        torch.tensor(
            [
                [True, True, False],
                [True, True, True],
            ],
            dtype=torch.bool,
        ),
    )
    assert torch.equal(loader.data_mask.cpu(), loader.listed_mask.cpu())
    assert torch.equal(loader.tradable_mask.cpu(), loader.listed_mask.cpu())
    assert torch.equal(loader.valid_mask.cpu(), loader.tradable_mask.cpu())
    assert torch.equal(loader.cs_mask.cpu(), loader.tradable_mask.cpu())

    feat = loader.feat_tensor[:, :, 0].cpu()
    assert torch.allclose(feat[0, :2], torch.tensor([0.0, 1.0], dtype=torch.float32))
    assert torch.isnan(feat[0, 2])
    assert torch.allclose(feat[1], torch.tensor([0.5, 1.0, 0.0], dtype=torch.float32))


def test_future_new_bond_does_not_change_history_cs_feature(monkeypatch):
    monkeypatch.setattr(
        "model_core.config.ModelConfig.INPUT_FEATURES",
        ["DBLOW_CS_RANK"],
        raising=False,
    )
    monkeypatch.setattr(
        "model_core.config.ModelConfig.WARMUP_DAYS",
        0,
        raising=False,
    )

    base_df = _build_minimal_cb_df(include_future_bond=False)
    future_df = _build_minimal_cb_df(include_future_bond=True)

    monkeypatch.setattr(
        "model_core.data_loader.pd.read_parquet",
        lambda _: base_df.copy(),
    )
    base_loader = CBDataLoader()
    base_loader.load_data(start_date="2026-01-02")

    monkeypatch.setattr(
        "model_core.data_loader.pd.read_parquet",
        lambda _: future_df.copy(),
    )
    future_loader = CBDataLoader()
    future_loader.load_data(start_date="2026-01-02")

    base_hist = base_loader.feat_tensor[0, :, 0].cpu()
    future_hist = future_loader.feat_tensor[0, :2, 0].cpu()

    assert torch.allclose(base_hist, future_hist, atol=1e-6)
    assert torch.equal(
        future_loader.cs_mask[0].cpu(),
        torch.tensor([True, True, False], dtype=torch.bool),
    )
