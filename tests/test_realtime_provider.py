"""
RealtimeDataProvider 单元测试

测试数据适配器的核心功能:
1. QMT 数据格式化
2. SQL 数据获取
3. feat_tensor 构建
"""
import pytest
import pandas as pd
import numpy as np
import torch
from unittest.mock import MagicMock, patch
from datetime import datetime

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_pipeline.realtime_provider import RealtimeDataProvider


class TestRealtimeDataProvider:
    """测试 RealtimeDataProvider"""
    
    @pytest.fixture
    def mock_sql_engine(self):
        """创建模拟 SQL 引擎"""
        engine = MagicMock()
        return engine
    
    @pytest.fixture
    def provider(self, mock_sql_engine):
        """创建 Provider 实例"""
        return RealtimeDataProvider(sql_engine=mock_sql_engine)
    
    def test_format_qmt_data_basic(self, provider):
        """测试 QMT 数据格式化 - 基础场景"""
        # 模拟 QMT 返回数据
        mock_data = {
            '123001.SZ': pd.DataFrame({
                'open': [100.0, 101.0],
                'high': [102.0, 103.0],
                'low': [99.0, 100.0],
                'close': [101.0, 102.0],
                'volume': [1000, 1200],
                'amount': [100000, 120000],
            }, index=pd.to_datetime(['2026-02-07', '2026-02-08'])),
            '127050.SZ': pd.DataFrame({
                'open': [110.0, 111.0],
                'high': [112.0, 113.0],
                'low': [109.0, 110.0],
                'close': [111.0, 112.0],
                'volume': [2000, 2200],
                'amount': [220000, 242000],
            }, index=pd.to_datetime(['2026-02-07', '2026-02-08'])),
        }
        
        code_list = ['123001.SZ', '127050.SZ']
        result = provider._format_qmt_data(mock_data, code_list)
        
        # 验证结果
        assert len(result) == 2
        assert 'code' in result.columns
        assert 'open' in result.columns
        assert 'high' in result.columns
        assert 'close' in result.columns
        assert 'vol' in result.columns
        
        # 验证取的是最后一行数据
        row_1 = result[result['code'] == '123001.SZ'].iloc[0]
        assert row_1['close'] == 102.0
        assert row_1['vol'] == 1200
    
    def test_format_qmt_data_missing_code(self, provider):
        """测试 QMT 数据格式化 - 缺失标的"""
        mock_data = {
            '123001.SZ': pd.DataFrame({
                'open': [100.0],
                'high': [102.0],
                'low': [99.0],
                'close': [101.0],
                'volume': [1000],
                'amount': [100000],
            }, index=pd.to_datetime(['2026-02-08'])),
        }
        
        code_list = ['123001.SZ', '127050.SZ']  # 127050 不在数据中
        result = provider._format_qmt_data(mock_data, code_list)
        
        # 只应返回存在的标的
        assert len(result) == 1
        assert result.iloc[0]['code'] == '123001.SZ'
    
    def test_format_qmt_data_empty(self, provider):
        """测试 QMT 数据格式化 - 空数据"""
        mock_data = {}
        code_list = ['123001.SZ']
        result = provider._format_qmt_data(mock_data, code_list)
        
        assert len(result) == 0
    
    def test_build_feat_tensor_basic(self, provider):
        """测试 feat_tensor 构建 - 基础场景"""
        # 模拟实时行情
        realtime_quotes = pd.DataFrame({
            'code': ['123001.SZ', '127050.SZ'],
            'trade_date': ['2026-02-08', '2026-02-08'],
            'open': [100.0, 110.0],
            'high': [102.0, 112.0],
            'low': [99.0, 109.0],
            'close': [101.0, 111.0],
            'vol': [1000.0, 2000.0],
        })


        
        # 模拟 CB 特性数据 (使用实际字段名)
        cb_features = pd.DataFrame({
            'code': ['123001.SZ', '127050.SZ'],
            'name': ['测试转债1', '测试转债2'],
            'trade_date': ['2026-02-08', '2026-02-08'],
            'close': [100.0, 109.0],  # 将被实时数据覆盖
            'open': [99.0, 108.0],    # 将被实时数据覆盖
            'high': [101.0, 110.0],   # 将被实时数据覆盖
            'vol': [800.0, 1800.0],   # 将被实时数据覆盖
            'conv_prem': [0.1, 0.15],
            'dblow': [120.0, 130.0],
            'remain_size': [5.0, 8.0],
            'pct_chg': [0.01, 0.02],
            'pct_chg_5': [0.05, 0.08],
            'volatility_stk': [0.3, 0.4],
            'pct_chg_stk': [0.015, 0.025],
            'pct_chg_5_stk': [0.06, 0.09],
            'pure_value': [95.0, 92.0],
            'alpha_pct_chg_5': [-0.01, -0.01],
            'cap_mv_rate': [0.05, 0.08],
            'turnover': [0.02, 0.03],
            'IV': [0.25, 0.30],
            'stock_vol60d': [0.28, 0.35],
            'convprem_zscore': [0.5, 1.0],
            'left_years': [3.5, 4.2],
            'list_days': [10, 12],
        })
        
        with patch.object(provider, 'sql_engine'):
            result = provider.build_feat_tensor(realtime_quotes, cb_features)
        
        # 验证维度: [Assets=2, Features=N]
        assert result.shape[0] == 2
        assert result.shape[1] > 0
        assert isinstance(result, torch.Tensor)
    
    def test_build_feat_tensor_empty_realtime(self, provider):
        """测试 feat_tensor 构建 - 无实时数据"""
        realtime_quotes = pd.DataFrame()  # 空
        
        cb_features = pd.DataFrame({
            'code': ['123001.SZ'],
            'name': ['测试转债1'],
            'trade_date': ['2026-02-08'],
            'close': [100.0],
            'open': [99.0],
            'high': [101.0],
            'vol': [800.0],
            'conv_prem': [0.1],
            'dblow': [120.0],
            'remain_size': [5.0],
            'pct_chg': [0.01],
            'pct_chg_5': [0.05],
            'volatility_stk': [0.3],
            'pct_chg_stk': [0.015],
            'pct_chg_5_stk': [0.06],
            'pure_value': [95.0],
            'alpha_pct_chg_5': [-0.01],
            'cap_mv_rate': [0.05],
            'turnover': [0.02],
            'IV': [0.25],
            'stock_vol60d': [0.28],
            'convprem_zscore': [0.5],
            'left_years': [3.5],
            'list_days': [10],
        })
        
        result = provider.build_feat_tensor(realtime_quotes, cb_features)
        
        # 应该使用 SQL 数据
        assert result.shape[0] == 1
    
    def test_get_asset_list(self, provider):
        """测试资产列表获取"""
        cb_features = pd.DataFrame({
            'code': ['123001.SZ', '127050.SZ', '128001.SZ'],
            'name': ['A', 'B', 'C'],
        })
        
        result = provider.get_asset_list(cb_features)
        
        assert result == ['123001.SZ', '127050.SZ', '128001.SZ']
    
    def test_get_names_dict(self, provider):
        """测试名称字典获取"""
        cb_features = pd.DataFrame({
            'code': ['123001.SZ', '127050.SZ'],
            'name': ['包钢转债', '蓝帆转债'],
        })
        
        result = provider.get_names_dict(cb_features)

        assert result['123001.SZ'] == '包钢转债'
        assert result['127050.SZ'] == '蓝帆转债'

    def test_tradable_mask_excludes_short_list_days(self, provider):
        raw_tensors = {
            'CLOSE': torch.tensor([[100.0, 110.0]], dtype=torch.float32),
            'VOL': torch.tensor([[1000.0, 1000.0]], dtype=torch.float32),
            'LEFT_YRS': torch.tensor([[2.0, 2.0]], dtype=torch.float32),
            'LIST_DAYS': torch.tensor([[3.0, 2.0]], dtype=torch.float32),
        }

        mask = provider._build_tradable_mask_from_raw_tensors(raw_tensors)

        assert torch.equal(mask.cpu(), torch.tensor([[True, False]], dtype=torch.bool))


class TestRealtimeDataProviderXtdata:
    """测试 xtdata 相关功能 (需要 Mock)"""
    
    @pytest.fixture
    def provider_with_mock_xtdata(self):
        """创建带有 Mock xtdata 的 Provider"""
        provider = RealtimeDataProvider(sql_engine=MagicMock())
        
        # Mock xtdata
        mock_xtdata = MagicMock()
        mock_xtdata.download_history_data = MagicMock()
        mock_xtdata.subscribe_quote = MagicMock()
        # Mock get_full_tick (新实现使用的接口)
        mock_xtdata.get_full_tick = MagicMock(return_value={
            '123001.SZ': {
                'timetag': '2026-02-08 14:50:00',
                'lastPrice': 101.5,
                'open': 100.0,
                'high': 102.0,
                'low': 99.0,
                'lastClose': 100.5,
                'volume': 1000,
                'amount': 100000,
                'bidPrice': [101.4, 101.3],
                'askPrice': [101.6, 101.7],
            },
        })
        # Mock get_market_data_ex (备用 K 线接口)
        mock_xtdata.get_market_data_ex = MagicMock(return_value={
            '123001.SZ': pd.DataFrame({
                'open': [100.0],
                'high': [102.0],
                'low': [99.0],
                'close': [101.0],
                'volume': [1000],
                'amount': [100000],
            }, index=pd.to_datetime(['2026-02-08'])),
        })
        
        provider._xtdata = mock_xtdata
        return provider
    
    def test_download_history_data(self, provider_with_mock_xtdata):
        """测试历史数据下载"""
        provider = provider_with_mock_xtdata
        code_list = ['123001.SZ', '127050.SZ']
        
        provider.download_history_data(code_list)
        
        # 验证调用次数
        assert provider._xtdata.download_history_data.call_count == 2
    
    def test_subscribe_quotes(self, provider_with_mock_xtdata):
        """测试行情订阅"""
        provider = provider_with_mock_xtdata
        code_list = ['123001.SZ']
        
        provider.subscribe_quotes(code_list)
        
        provider._xtdata.subscribe_quote.assert_called_once()
    
    def test_get_realtime_quotes(self, provider_with_mock_xtdata):
        """测试实时行情获取 (基于 get_full_tick)"""
        provider = provider_with_mock_xtdata
        code_list = ['123001.SZ']
        
        result = provider.get_realtime_quotes(code_list)
        
        # 验证调用了 get_full_tick 而非 get_market_data_ex
        provider._xtdata.get_full_tick.assert_called_once_with(code_list)
        
        assert len(result) == 1
        assert result.iloc[0]['code'] == '123001.SZ'
        # lastPrice 映射为 close
        assert result.iloc[0]['close'] == 101.5
        assert result.iloc[0]['open'] == 100.0
        assert result.iloc[0]['high'] == 102.0
        # 验证 trade_date 从 timetag 正确解析
        assert result.iloc[0]['trade_date'] == '2026-02-08'
        # 验证下游所需的全部列都存在
        for col in ['code', 'trade_date', 'open', 'high', 'low', 'close', 'vol', 'amount']:
            assert col in result.columns, f"Missing column: {col}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
