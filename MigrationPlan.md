# AlphaGPT A股/可转债改造迁移计划

## 概述

本文档记录将 AlphaGPT 从加密货币（Solana/Meme）因子挖掘框架改造为 A 股和可转债量化框架的详细技术方案。改造核心包括数据源重构、因子体系重设计、回测逻辑适配、执行层替换及风险管理升级。

---

## 1. 现状分析

### 1.1 现有架构

```
AlphaGPT/
├── model_core/           # 核心模型与因子引擎
│   ├── alphagpt.py       # Transformer模型
│   ├── engine.py         # 训练引擎
│   ├── factors.py        # 特征工程
│   ├── data_loader.py    # 数据加载
│   ├── backtest.py       # 回测引擎
│   ├── vm.py             # 公式虚拟机
│   └── ops.py            # 算子定义
├── data_pipeline/        # 数据管道
│   ├── fetcher.py        # 数据获取
│   ├── db_manager.py     # 数据库管理
│   ├── config.py         # 配置
│   └── providers/        # 数据源
├── execution/            # 执行层
│   ├── trader.py         # Solana交易
│   ├── jupiter.py        # DEX聚合
│   ├── rpc_handler.py    # RPC客户端
│   └── config.py         # 配置
├── strategy_manager/     # 策略管理
│   ├── runner.py         # 策略运行器
│   ├── portfolio.py      # 组合管理
│   ├── risk.py           # 风险管理
│   └── config.py         # 配置
└── dashboard/            # 可视化
```

### 1.2 当前数据模型

**加密货币特有字段**：
- `address` - Token 地址
- `liquidity` - 池子流动性（USD）
- `fdv` - 完全稀释估值
- 交易机制：T+0、无涨跌停、无限流动性

---

## 2. 改造目标

### 2.1 功能目标

| 目标 | 说明 |
|------|------|
| A股全市场覆盖 | 支持 5000+ 只股票的因子挖掘 |
| 可转债专项因子 | 转股溢价率、双低因子、强赎监测 |
| T+1 交易模拟 | 准确还原 A 股交易制度 |
| 多周期支持 | 日线/周线/月线级别因子 |
| 行业/板块因子 | 加入行业分类、市值因子 |

### 2.2 技术目标

- 数据延迟：分钟级（盘中）/ 日终（盘后）
- 回测精度：逐笔/_tick 级（可选）
- 实盘对接：支持华泰/中信/银河等主流券商

---

## 3. 数据层改造

### 3.1 数据源选择

#### 3.1.1 A股数据源

| 数据源 | 类型 | 覆盖 | 成本 | 推荐度 |
|--------|------|------|------|--------|
| Tushare Pro | API | 全市场 | 免费/付费 | ⭐⭐⭐⭐⭐ |
| 聚宽 (JoinQuant) | API | 全市场 | 免费/付费 | ⭐⭐⭐⭐ |
| 米筐 (RiceQuant) | API | 全市场 | 付费 | ⭐⭐⭐⭐ |
| akshare | 开源 | 全市场 | 免费 | ⭐⭐⭐ |

**推荐方案**：Tushare Pro（数据质量好，文档完善）

```python
# data_pipeline/providers/tushare.py
import tushare as ts
from datetime import datetime, timedelta

class TushareProvider:
    def __init__(self, token: str):
        self.pro = ts.pro_api(token)
    
    async def get_daily(self, trade_date: str = None):
        """获取日线数据"""
        df = self.pro.daily(
            trade_date=trade_date or datetime.now().strftime('%Y%m%d')
        )
        return df
    
    async def get_adj_factor(self, ts_code: str, start: str, end: str):
        """获取复权因子"""
        df = self.pro.adj_factor(
            ts_code=ts_code,
            start_date=start,
            end_date=end
        )
        return df
```

#### 3.1.2 可转债数据源

| 数据源 | 说明 |
|--------|------|
| 集思录 | 官方可转债数据，实时性好 |
| 东方财富 | 可转债页面数据 |
| Wind | 机构级数据，费用高 |
| akshare | 集思录数据接口 |

```python
# data_pipeline/providers/jisilu.py (推荐)
import akshare as ak

class JisiluProvider:
    async def get_cb_list(self):
        """可转债列表"""
        return ak.bond_zh_hs_cov()
    
    async def get_cb_info(self, bond_id: str):
        """可转债详情"""
        return ak.bond_zh_hs_cov_detail(symbol=bond_id)
```

### 3.2 数据模型重构

#### 3.2.1 A股数据结构

```python
# data_pipeline/schema.py
from dataclasses import dataclass
from typing import Optional
from datetime import datetime

@dataclass
class AStockDaily:
    ts_code: str              # 股票代码，如 '600519.SH'
    trade_date: str           # 交易日期 YYYYMMDD
    open: float               # 开盘价
    high: float               # 最高价
    low: float                # 最低价
    close: float              # 收盘价
    pre_close: float          # 昨收价
    change: float             # 涨跌额
    pct_chg: float            # 涨跌幅
    vol: float                # 成交量（手）
    amount: float             # 成交额（千元）
    turnover_rate: float      # 换手率
    is_halt: int              # 是否停牌 0/1
    
    @property
    def is_limit_up(self) -> bool:
        return self.pct_chg >= 9.5
    
    @property
    def is_limit_down(self) -> bool:
        return self.pct_chg <= -9.5
```

#### 3.2.2 可转债数据结构

```python
@dataclass
class ConvertibleBond:
    bond_id: str              # 债券代码，如 '113050.SH'
    bond_name: str            # 债券简称
    stock_code: str           # 正股代码
    stock_name: str           # 正股简称
    convert_price: float      # 转股价
    conv_ratio: float         # 转股比例
    conv_value: float         # 转股价值
    premium_rate: float       # 转股溢价率
    remain_amount: float      # 剩余规模（亿元）
    maturity_date: str        # 到期日
    issue_date: str           # 发行日
    delist_date: Optional[str]# 强赎/退市日期
    ytm: float                # 到期收益率
    cb_price: float           # 可转债价格
    cb_chg: float             # 可转债涨跌幅
    
    @property
    def is_strong_redeem_risk(self) -> bool:
        """强赎风险检测：转股价值>130连续15天"""
        return self.conv_value > 130
```

### 3.3 数据库表结构

```sql
-- PostgreSQL DDL

-- A股基础信息表
CREATE TABLE a_stocks (
    ts_code VARCHAR(20) PRIMARY KEY,
    name VARCHAR(50) NOT NULL,
    industry VARCHAR(50),
    area VARCHAR(20),
    market VARCHAR(20),           -- 主板/中小板/创业板/科创板
    list_date DATE,
    delist_date DATE,
    is_halt BOOLEAN DEFAULT FALSE,
    last_updated TIMESTAMP DEFAULT NOW()
);

-- A股日线数据表
CREATE TABLE a_stock_daily (
    id BIGSERIAL PRIMARY KEY,
    ts_code VARCHAR(20) NOT NULL,
    trade_date DATE NOT NULL,
    open_price DECIMAL(10, 3),
    high_price DECIMAL(10, 3),
    low_price DECIMAL(10, 3),
    close_price DECIMAL(10, 3),
    pre_close DECIMAL(10, 3),
    change_price DECIMAL(10, 3),
    pct_chg DECIMAL(8, 4),
    vol BIGINT,                    -- 手
    amount DECIMAL(16, 4),         -- 千元
    turnover_rate DECIMAL(8, 4),
    is_halt SMALLINT DEFAULT 0,
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(ts_code, trade_date)
);

-- 可转债基础信息表
CREATE TABLE convertible_bonds (
    bond_id VARCHAR(20) PRIMARY KEY,
    bond_name VARCHAR(50) NOT NULL,
    stock_code VARCHAR(20) NOT NULL,
    stock_name VARCHAR(50),
    convert_price DECIMAL(10, 4),
    conv_ratio DECIMAL(10, 6),
    issue_amount DECIMAL(14, 4),   -- 发行规模（亿元）
    remain_amount DECIMAL(14, 4),  -- 剩余规模
    maturity_date DATE,
    issue_date DATE,
    delist_date DATE,
    is_listed BOOLEAN DEFAULT TRUE,
    last_updated TIMESTAMP DEFAULT NOW()
);

-- 可转债日线数据表
CREATE TABLE cb_daily (
    id BIGSERIAL PRIMARY KEY,
    bond_id VARCHAR(20) NOT NULL,
    trade_date DATE NOT NULL,
    open_price DECIMAL(10, 3),
    high_price DECIMAL(10, 3),
    low_price DECIMAL(10, 3),
    close_price DECIMAL(10, 3),
    pre_close DECIMAL(10, 3),
    change_price DECIMAL(10, 4),
    pct_chg DECIMAL(8, 4),
    vol BIGINT,
    amount DECIMAL(16, 4),
    conv_value DECIMAL(10, 4),     -- 转股价值
    premium_rate DECIMAL(8, 4),    -- 转股溢价率
    ytm DECIMAL(8, 4),             -- 到期收益率
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(bond_id, trade_date)
);

-- 创建索引
CREATE INDEX idx_a_stock_daily_date ON a_stock_daily(trade_date);
CREATE INDEX idx_a_stock_daily_tscode ON a_stock_daily(ts_code);
CREATE INDEX idx_cb_daily_date ON cb_daily(trade_date);
CREATE INDEX idx_cb_daily_bondid ON cb_daily(bond_id);
CREATE INDEX idx_cb_link_stock ON convertible_bonds(stock_code);

-- 行业分类表（可选）
CREATE TABLE industry分类 (
    ts_code VARCHAR(20) PRIMARY KEY,
    industry VARCHAR(50),
    sub_industry VARCHAR(50),
    concept_tags TEXT[]
);
```

### 3.4 数据加载器改造

```python
# model_core/data_loader.py 改造

import pandas as pd
import torch
import sqlalchemy
from .config import ModelConfig
from .factors import FeatureEngineer
from typing import Optional, Dict, List

class AStockDataLoader:
    """A股数据加载器"""
    
    def __init__(self, db_url: str = None):
        self.engine = sqlalchemy.create_engine(db_url or ModelConfig.DB_URL)
        self.feat_tensor: Optional[torch.Tensor] = None
        self.raw_data_cache: Optional[Dict[str, torch.Tensor]] = None
        self.target_ret: Optional[torch.Tensor] = None
        self.stock_list: List[str] = []
    
    def load_data(
        self, 
        start_date: str, 
        end_date: str,
        stock_list: List[str] = None,
        limit: int = 2000
    ):
        """加载A股数据"""
        print(f"Loading A-Stock data: {start_date} ~ {end_date}")
        
        # 查询股票列表
        if stock_list is None:
            stock_list = self._get_stock_list(limit)
        self.stock_list = stock_list
        
        if not stock_list:
            raise ValueError("No stocks found.")
        
        # 转换日期格式
        start = pd.to_datetime(start_date).strftime('%Y%m%d')
        end = pd.to_datetime(end_date).strftime('%Y%m%d')
        
        # 查询日线数据
        ts_code_placeholders = "'" + "','".join(stock_list) + "'"
        query = f"""
            SELECT ts_code, trade_date, open, high, low, close, 
                   pre_close, pct_chg, vol, amount, turnover_rate
            FROM a_stock_daily
            WHERE ts_code IN ({ts_code_placeholders})
            AND trade_date BETWEEN '{start}' AND '{end}'
            ORDER BY trade_date ASC
        """
        
        df = pd.read_sql(query, self.engine)
        
        # 数据透视与预处理
        self.raw_data_cache = self._pivot_to_tensors(df)
        
        # 特征工程
        self.feat_tensor = FeatureEngineer.compute_astock_features(
            self.raw_data_cache
        )
        
        # 计算未来收益（用于训练）
        close = self.raw_data_cache['close']
        t1 = torch.roll(close, -1, dims=1)
        t5 = torch.roll(close, -5, dims=1)
        self.target_ret = torch.log(t5 / (t1 + 1e-9))
        self.target_ret[:, -5:] = 0.0
        
        print(f"A-Stock Data Ready. Shape: {self.feat_tensor.shape}")
    
    def _pivot_to_tensors(self, df: pd.DataFrame) -> Dict[str, torch.Tensor]:
        """将DataFrame透视为张量"""
        def to_tensor(column: str) -> torch.Tensor:
            pivot = df.pivot(
                index='trade_date', 
                columns='ts_code', 
                values=column
            )
            pivot = pivot.sort_index()
            # 前向填充处理停牌
            pivot = pivot.ffill().fillna(0.0)
            return torch.tensor(
                pivot.values.T, 
                dtype=torch.float32, 
                device=ModelConfig.DEVICE
            )
        
        return {
            'open': to_tensor('open'),
            'high': to_tensor('high'),
            'low': to_tensor('low'),
            'close': to_tensor('close'),
            'pre_close': to_tensor('pre_close'),
            'pct_chg': to_tensor('pct_chg'),
            'vol': to_tensor('vol'),
            'amount': to_tensor('amount'),
            'turnover_rate': to_tensor('turnover_rate')
        }
    
    def _get_stock_list(self, limit: int) -> List[str]:
        """获取股票列表"""
        query = f"""
            SELECT ts_code FROM a_stocks 
            WHERE is_halt = FALSE
            ORDER BY market_cap DESC
            LIMIT {limit}
        """
        df = pd.read_sql(query, self.engine)
        return df['ts_code'].tolist()


class CBDataLoader(StockDataLoader):
    """可转债数据加载器（继承自A股）"""
    
    def __init__(self, db_url: str = None):
        super().__init__(db_url)
        self.bond_list: List[str] = []
    
    def load_data(
        self,
        start_date: str,
        end_date: str,
        bond_list: List[str] = None,
        limit: int = 500
    ):
        """加载可转债数据"""
        print(f"Loading CB data: {start_date} ~ {end_date}")
        
        self.bond_list = bond_list or self._get_bond_list(limit)
        
        # 可转债数据查询
        ts_code_placeholders = "'" + "','".join(self.bond_list) + "'"
        query = f"""
            SELECT bd.bond_id, trade_date, open, high, low, close,
                   pre_close, pct_chg, vol, amount,
                   conv_value, premium_rate, ytm
            FROM cb_daily bd
            WHERE bd.bond_id IN ({ts_code_placeholders})
            AND trade_date BETWEEN '{start_date}' AND '{end_date}'
            ORDER BY trade_date ASC
        """
        
        df = pd.read_sql(query, self.engine)
        
        self.raw_data_cache = self._pivot_to_tensors(df)
        
        # 可转债特征工程
        self.feat_tensor = FeatureEngineer.compute_cb_features(
            self.raw_data_cache
        )
        
        print(f"CB Data Ready. Shape: {self.feat_tensor.shape}")
    
    def get_linked_stock_features(self) -> Dict[str, torch.Tensor]:
        """获取对应正股的特征（用于可转债分析）"""
        # 关联查询正股数据
        query = f"""
            SELECT cb.stock_code, ts_code 
            FROM convertible_bonds cb
            WHERE cb.bond_id IN ('{"','".join(self.bond_list)}')
        """
        df = pd.read_sql(query, self.engine)
        # 加载正股日线数据...
        return {}
```

---

## 4. 因子体系重构

### 4.1 因子分类

#### 4.1.1 A股因子库

```python
# model_core/factors.py 新增

import torch
import torch.nn.functional as F
from typing import Dict

class AStockIndicators:
    """A股技术因子"""
    
    @staticmethod
    def momentum(close: torch.Tensor, n: int = 20) -> torch.Tensor:
        """动量因子：过去n日累计收益"""
        ret = torch.log(close / torch.roll(close, 1, dims=1))
        return ret.cumsum(dim=1)[:, -n:].mean(dim=1, keepdim=True)
    
    @staticmethod
    def volatility(close: torch.Tensor, n: int = 10) -> torch.Tensor:
        """波动率因子：收益标准差"""
        ret = torch.log(close / torch.roll(close, 1, dims=1))
        return ret.unfold(1, n, 1).std(dim=-1)
    
    @staticmethod
    def turnover_ratio(vol: torch.Tensor, market_cap: torch.Tensor) -> torch.Tensor:
        """换手率因子"""
        return vol / (market_cap + 1e-9)
    
    @staticmethod
    def reversal(close: torch.Tensor, n: int = 5) -> torch.Tensor:
        """反转因子：短期反转"""
        return -torch.log(close / torch.roll(close, n, dims=1))
    
    @staticmethod
    def amihud(close: torch.Tensor, vol: torch.Tensor, amount: torch.Tensor) -> torch.Tensor:
        """Amihud流动性比率：收益/成交额"""
        ret = torch.abs(torch.log(close / (torch.roll(close, 1, dims=1) + 1e-9)))
        return (ret * 1e6) / (amount + 1)  # 放大以便归一化
    
    @staticmethod
    def high_low_range(high: torch.Tensor, low: torch.Tensor, n: int = 10) -> torch.Tensor:
        """价格区间因子"""
        ranges = (high - low) / (low + 1e-9)
        return ranges.unfold(1, n, 1).mean(dim=-1, keepdim=True)
    
    @staticmethod
    def price_ema(close: torch.Tensor, span: int = 12) -> torch.Tensor:
        """EMA偏离"""
        ema = close.unfold(1, span, 1).mean(dim=-1)
        return (close[:, -1:] - ema) / (ema + 1e-9)
    
    @staticmethod
    def volume_ema(vol: torch.Tensor, span: int = 12) -> torch.Tensor:
        """量能偏离"""
        ema = vol.unfold(1, span, 1).mean(dim=-1)
        return (vol[:, -1:] - ema) / (ema + 1e-9)
    
    @staticmethod
    def rsi(close: torch.Tensor, period: int = 14) -> torch.Tensor:
        """RSI指标"""
        delta = close - torch.roll(close, 1, dims=1)
        delta[:, 0] = 0
        gain = delta.clamp(min=0)
        loss = (-delta).clamp(min=0)
        avg_gain = gain.unfold(1, period, 1).mean(dim=-1)
        avg_loss = loss.unfold(1, period, 1).mean(dim=-1)
        rs = avg_gain / (avg_loss + 1e-9)
        return 100 - 100 / (1 + rs)
    
    @staticmethod
    def macd(close: torch.Tensor, fast: int = 12, slow: int = 26) -> torch.Tensor:
        """MACD"""
        ema_fast = close.unfold(1, fast, 1).mean(dim=-1)
        ema_slow = close.unfold(1, slow, 1).mean(dim=-1)
        return ema_fast - ema_slow
    
    @staticmethod
    def boll(close: torch.Tensor, window: int = 20, k: float = 2.0) -> torch.Tensor:
        """布林带偏离"""
        mean = close.unfold(1, window, 1).mean(dim=-1)
        std = close.unfold(1, window, 1).std(dim=-1)
        return (close[:, -1:] - mean) / (k * std + 1e-9)


class MarketIndicators:
    """市场/行业因子"""
    
    @staticmethod
    def industry_momentum(close: torch.Tensor, industry_codes: torch.Tensor) -> torch.Tensor:
        """行业动量"""
        # 需要行业分类数据支持
        pass
    
    @staticmethod
    def size_factor(market_cap: torch.Tensor) -> torch.Tensor:
        """市值因子（大小盘）"""
        return torch.log(market_cap / market_cap.median())
    
    @staticmethod
    def value_factor(pe: torch.Tensor, pb: torch.Tensor) -> torch.Tensor:
        """价值因子"""
        return -torch.log(pe + 1) - torch.log(pb + 1)


class FundamentalIndicators:
    """基本面因子（需要单独获取）"""
    
    @staticmethod
    def pe_ratio(price: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        """PE"""
        return price / (eps + 1e-9)
    
    @staticmethod
    def pb_ratio(price: torch.Tensor, bvps: torch.Tensor) -> torch.Tensor:
        """PB"""
        return price / (bvps + 1e-9)
    
    @staticmethod
    def roe(net_profit: torch.Tensor, equity: torch.Tensor) -> torch.Tensor:
        """ROE"""
        return net_profit / (equity + 1e-9)
    
    @staticmethod
    def growth_factor(revenue: torch.Tensor, n: int = 4) -> torch.Tensor:
        """营收增速"""
        rev_change = revenue / (torch.roll(revenue, n, dims=1) + 1e-9)
        return torch.log(rev_change)
```

#### 4.1.2 可转债因子

```python
class CBIndicators:
    """可转债特有因子"""
    
    @staticmethod
    def conversion_premium(conv_value: torch.Tensor, price: torch.Tensor) -> torch.Tensor:
        """转股溢价率"""
        return price / conv_value - 1
    
    @staticmethod
    def double_low(price: torch.Tensor, premium: torch.Tensor) -> torch.Tensor:
        """双低因子：价格低 + 溢价率低"""
        normalized_price = (price - 100) / 100  # 相对面值归一化
        return normalized_price + premium.abs()
    
    @staticmethod
    def ytm_factor(ytm: torch.Tensor) -> torch.Tensor:
        """到期收益率因子"""
        return torch.clamp(ytm, -5, 15) / 15
    
    @staticmethod
    def conversion_value_ratio(conv_value: torch.Tensor, price: torch.Tensor) -> torch.Tensor:
        """转股价值/价格"""
        return conv_value / (price + 1e-9)
    
    @staticmethod
    def remain_scale_factor(remain_amount: torch.Tensor) -> torch.Tensor:
        """剩余规模因子（规模太小有流动性风险）"""
        return torch.log(remain_amount + 1)
    
    @staticmethod
    def redemption_risk(conv_value: torch.Tensor, threshold: float = 130.0) -> torch.Tensor:
        """强赎风险因子"""
        return (conv_value > threshold).float()
    
    @staticmethod
    def put_redemption_risk(cb_price: torch.Tensor, remain_amount: torch.Tensor) -> torch.Tensor:
        """回售风险因子（价格低于面值+剩余规模大）"""
        below_par = (cb_price < 100).float()
        large_scale = (remain_amount > 10).float()
        return below_par * large_scale
    
    @staticmethod
    def cb_momentum(cb_price: torch.Tensor, n: int = 20) -> torch.Tensor:
        """可转债动量"""
        return torch.log(cb_price / (torch.roll(cb_price, n, dims=1) + 1e-9))
    
    @staticmethod
    def stock_correlation(cb_price: torch.Tensor, stock_price: torch.Tensor) -> torch.Tensor:
        """可转债与正股相关性"""
        cb_ret = torch.log(cb_price / (torch.roll(cb_price, 1, dims=1) + 1e-9))
        stock_ret = torch.log(stock_price / (torch.roll(stock_price, 1, dims=1) + 1e-9))
        # 简化的相关性计算
        return (cb_ret * stock_ret).mean(dim=1, keepdim=True)
```

### 4.2 特征工程统一接口

```python
class FeatureEngineer:
    """统一的特征工程入口"""
    
    INPUT_DIM_ASTOCK = 12
    INPUT_DIM_CB = 18
    
    @staticmethod
    def compute_astock_features(raw_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """A股特征计算"""
        c = raw_dict['close']
        o = raw_dict['open']
        h = raw_dict['high']
        l = raw_dict['low']
        v = raw_dict['vol']
        amt = raw_dict['amount']
        tr = raw_dict['turnover_rate']
        pct = raw_dict['pct_chg']
        
        # 收益因子
        ret = torch.log(c / (torch.roll(c, 1, dims=1) + 1e-9))
        
        # 动量
        mom_5 = AStockIndicators.momentum(c, 5)
        mom_20 = AStockIndicators.momentum(c, 20)
        
        # 波动率
        vol_10 = AStockIndicators.volatility(c, 10)
        
        # 流动性
        liq = AStockIndicators.amihud(c, v, amt)
        
        # 换手率
        turnover = tr.unsqueeze(1)
        
        # 反转
        rev_5 = AStockIndicators.reversal(c, 5)
        
        # 价格区间
        hl_range = AStockIndicators.high_low_range(h, l, 10)
        
        # RSI
        rsi = AStockIndicators.rsi(c)
        
        # MACD
        macd = AStockIndicators.macd(c)
        
        # 布林带
        boll = AStockIndicators.boll(c)
        
        def robust_norm(t: torch.Tensor) -> torch.Tensor:
            median = torch.nanmedian(t, dim=1, keepdim=True)[0]
            mad = torch.nanmedian(torch.abs(t - median), dim=1, keepdim=True)[0] + 1e-6
            return torch.clamp((t - median) / mad, -5.0, 5.0)
        
        features = torch.cat([
            robust_norm(ret.unsqueeze(1)),
            robust_norm(mom_5),
            robust_norm(mom_20),
            robust_norm(vol_10),
            robust_norm(liq),
            robust_norm(turnover),
            robust_norm(rev_5),
            robust_norm(hl_range),
            robust_norm(rsi),
            robust_norm(macd),
            robust_norm(boll),
            robust_norm(pct.unsqueeze(1))
        ], dim=1)
        
        return features
    
    @staticmethod
    def compute_cb_features(raw_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """可转债特征计算"""
        cb_price = raw_dict['close']
        conv_value = raw_dict['conv_value']
        premium = raw_dict['premium_rate']
        ytm = raw_dict['ytm']
        vol = raw_dict['vol']
        remain = raw_dict['remain_amount']
        
        # 可转债特有因子
        conv_premium = CBIndicators.conversion_premium(conv_value, cb_price)
        double_low = CBIndicators.double_low(cb_price, premium)
        ytm_f = CBIndicators.ytm_factor(ytm)
        conv_ratio = CBIndicators.conversion_value_ratio(conv_value, cb_price)
        scale = CBIndicators.remain_scale_factor(remain)
        redemption = CBIndicators.redemption_risk(conv_value)
        cb_mom = CBIndicators.cb_momentum(cb_price, 20)
        
        def robust_norm(t: torch.Tensor) -> torch.Tensor:
            median = torch.nanmedian(t, dim=1, keepdim=True)[0]
            mad = torch.nanmedian(torch.abs(t - median), dim=1, keepdim=True)[0] + 1e-6
            return torch.clamp((t - median) / mad, -5.0, 5.0)
        
        features = torch.cat([
            robust_norm((cb_price - 100).unsqueeze(1)),  # 价格偏离面值
            robust_norm(conv_premium),
            robust_norm(double_low),
            robust_norm(ytm_f),
            robust_norm(conv_ratio),
            robust_norm(scale),
            robust_norm(redemption),
            robust_norm(cb_mom),
        ], dim=1)
        
        return features
```

---

## 5. 回测引擎重构

### 5.1 A股回测逻辑

```python
# model_core/backtest.py 改造

import torch
from typing import Tuple, Dict

class AStockBacktest:
    """A股回测引擎"""
    
    def __init__(self):
        # 交易成本
        self.commission = 0.0003      # 万三佣金
        self.stamp_duty = 0.001        # 千一印花税（卖出收取）
        self.min_commission = 5.0      # 最低佣金5元
        
        # 交易规则
        self.t_plus_1 = True           # T+1交易
        self.limit_up_ratio = 0.095    # 涨停板（约10%，考虑四舍五入）
        self.limit_down_ratio = -0.095 # 跌停板
        
    def evaluate(
        self, 
        signals: torch.Tensor,
        raw_data: Dict[str, torch.Tensor],
        target_ret: torch.Tensor,
        position_ratio: float = 1.0
    ) -> Tuple[torch.Tensor, float]:
        """
        回测评估
        
        Args:
            signals: 策略信号 [N, T]
            raw_data: 原始数据字典
            target_ret: 目标收益 [N, T]
            position_ratio: 仓位比例
        
        Returns:
            fitness: 最终得分
            avg_ret: 平均收益
        """
        close = raw_data['close']
        n_stocks = close.shape[0]
        
        # 涨跌停检测
        limit_up = target_ret > self.limit_up_ratio
        limit_down = target_ret < self.limit_down_ratio
        
        # 生成交易信号
        position = (signals > 0.85).float() * position_ratio
        
        # 涨跌停过滤
        position = position * (~limit_up).float()  # 涨停无法买入
        position = position * (~limit_down).float()  # 跌停应卖出但已包含
        
        # T+1 限制：今日买入，次日才能卖出
        prev_position = torch.roll(position, 1, dims=1)
        prev_position[:, 0] = 0  # 第一天没有历史
        
        # 只能卖出昨天及更早买入的仓位
        sell_allowed = prev_position > 0
        actual_sell = (position < prev_position).float() * sell_allowed
        actual_buy = (position > prev_position).float()
        
        # 计算换手率
        turnover = actual_buy + actual_sell
        
        # 交易成本
        commission_cost = turnover * self.commission
        commission_cost = torch.maximum(
            commission_cost, 
            self.min_commission / close[:, -1:]  # 最小佣金约束
        )
        
        # 印花税（仅卖出收取）
        stamp_duty_cost = actual_sell * self.stamp_duty
        
        total_cost = commission_cost + stamp_duty_cost
        
        # 收益计算
        gross_pnl = position * target_ret
        net_pnl = gross_pnl - total_cost
        
        # 夏普比率（简化）
        daily_ret = net_pnl.mean(dim=1)
        sharpe = daily_ret.mean() / (daily_ret.std() + 1e-9)
        
        # 最大回撤
        cum_ret = net_pnl.cumsum(dim=1)
        running_max = torch.maximum.accumulate(cum_ret)
        drawdown = running_max - cum_ret
        max_drawdown = drawdown.max()
        
        # 综合得分
        score = sharpe * 2 + net_pnl.sum(dim=1).mean() * 10 - max_drawdown * 5
        
        return score.mean(), net_pnl.sum(dim=1).mean().item()
```

### 5.2 可转债回测逻辑

```python
class CBBacktest:
    """可转债回测引擎"""
    
    def __init__(self):
        self.commission = 0.0001       # 万一佣金（可转债费率更低）
        self.stamp_duty = 0.0          # 债券无印花税
        self.t_plus_0 = True           # T+0交易
        self.min_commission = 1.0      # 最低佣金1元
        
    def evaluate(
        self,
        signals: torch.Tensor,
        raw_data: Dict[str, torch.Tensor],
        target_ret: torch.Tensor
    ) -> Tuple[torch.Tensor, float]:
        """可转债回测"""
        cb_price = raw_data['close']
        conv_value = raw_data['conv_value']
        premium = raw_data['premium_rate']
        
        # 基础仓位
        position = (signals > 0.8).float()
        
        # 强赎风险过滤
        strong_redeem = conv_value > 130
        position = position * (~strong_redeem).float()
        
        # 双低约束：溢价率过高不买入
        high_premium = premium > 0.5  # 溢价率>50%
        position = position * (~high_premium).float()
        
        # 流动性过滤
        vol = raw_data['vol']
        low_liquidity = vol < 10000  # 成交量<1万
        position = position * (~low_liquidity).float()
        
        # 交易成本
        turnover = torch.abs(torch.roll(position, 1, dims=1) - position)
        turnover[:, 0] = position[:, 0]
        
        cost = turnover * self.commission
        cost = torch.maximum(cost, self.min_commission / cb_price[:, -1:])
        
        # 收益
        gross_pnl = position * target_ret
        net_pnl = gross_pnl - cost
        
        score = net_pnl.sum(dim=1).mean() * 10
        return score, net_pnl.sum(dim=1).mean().item()
```

---

## 6. 执行层改造

### 6.1 A股实盘对接

```python
# execution/a_stock_trader.py

import asyncio
from typing import Optional, Dict
from decimal import Decimal
from loguru import logger

class AStockTrader:
    """A股交易接口"""
    
    def __init__(self, broker: str = 'huatai'):
        """
        Args:
            broker: 券商选择 'huatai'/'zhongxin'/'guoyuan' 等
        """
        self.broker = broker
        self.position: Dict[str, Dict] = {}  # {ts_code: {qty, avg_price}}
        self.cash = 0.0
        
        if broker == 'huatai':
            self._init_huatai()
        elif broker == 'guotai_junan':
            self._init_gtja()
        else:
            self._init_simulate()
    
    def _init_huatai(self):
        """华泰证券API初始化"""
        # 实际接入需要使用华泰开放的API（如XTP或第三方封装）
        pass
    
    def _init_simulate(self):
        """模拟交易"""
        logger.warning("Using simulated trading mode")
    
    async def buy(
        self, 
        ts_code: str, 
        amount_cny: float,
        price_type: str = 'limit',
        limit_price: float = None
    ) -> Dict:
        """
        买入股票
        
        Args:
            ts_code: 股票代码，如 '600519.SH'
            amount_cny: 买入金额（元）
            price_type: 'market'/'limit'
            limit_price: 限价价格
        """
        # 计算买入股数（A股必须是100的整数倍）
        if limit_price:
            price = limit_price
        else:
            price = await self._get_market_price(ts_code)
        
        # 买入股数 = 金额 / 价格，取整为100的倍数
        qty = int(amount_cny / price / 100) * 100
        if qty < 100:
            logger.warning(f"Amount too small for {ts_code}, qty={qty}")
            return {'success': False, 'reason': 'qty_too_small'}
        
        # 模拟下单
        if self._is_simulated():
            return self._simulate_trade(ts_code, qty, price, 'buy')
        
        # 实际API调用
        try:
            result = await self._real_trade(ts_code, qty, price, 'buy')
            return result
        except Exception as e:
            logger.error(f"Buy order failed: {e}")
            return {'success': False, 'reason': str(e)}
    
    async def sell(
        self, 
        ts_code: str, 
        qty: int = None,
        percentage: float = 1.0,
        price_type: str = 'limit',
        limit_price: float = None
    ) -> Dict:
        """
        卖出股票
        
        Args:
            ts_code: 股票代码
            qty: 卖出股数，不指定则按percentage
            percentage: 卖出持仓比例
            price_type: 'market'/'limit'
            limit_price: 限价
        """
        if ts_code not in self.position:
            return {'success': False, 'reason': 'no_position'}
        
        current_qty = self.position[ts_code]['qty']
        sell_qty = qty or int(current_qty * percentage / 100) * 100
        sell_qty = min(sell_qty, current_qty)
        
        if sell_qty < 100:
            return {'success': False, 'reason': 'qty_too_small'}
        
        # T+1 检查
        if self._is_t_plus_1_violation(ts_code):
            return {'success': False, 'reason': 't_plus_1_violation'}
        
        price = limit_price or await self._get_market_price(ts_code)
        
        if self._is_simulated():
            return self._simulate_trade(ts_code, sell_qty, price, 'sell')
        
        return await self._real_trade(ts_code, sell_qty, price, 'sell')
    
    async def get_positions(self) -> Dict[str, Dict]:
        """获取持仓"""
        if self._is_simulated():
            return self.position.copy()
        # 实际API调用
        return await self._fetch_real_positions()
    
    async def get_cash(self) -> float:
        """获取可用资金"""
        return self.cash
    
    def _is_simulated(self) -> bool:
        return self.broker == 'simulate'
    
    def _is_t_plus_1_violation(self, ts_code: str) -> bool:
        """T+1 检查"""
        if ts_code not in self.position:
            return False
        pos = self.position[ts_code]
        # 检查是否今日买入
        if pos.get('buy_today', False):
            return True
        return False
    
    def _simulate_trade(
        self, 
        ts_code: str, 
        qty: int, 
        price: float, 
        side: str
    ) -> Dict:
        """模拟交易"""
        if side == 'buy':
            cost = qty * price * (1 + 0.0003)
            if cost > self.cash:
                return {'success': False, 'reason': 'insufficient_funds'}
            
            self.cash -= cost
            
            if ts_code in self.position:
                pos = self.position[ts_code]
                total_qty = pos['qty'] + qty
                avg_price = (pos['qty'] * pos['avg_price'] + qty * price) / total_qty
                pos['qty'] = total_qty
                pos['avg_price'] = avg_price
            else:
                self.position[ts_code] = {
                    'qty': qty,
                    'avg_price': price,
                    'buy_today': True  # 标记今日买入
                }
            
            logger.info(f"[SIM] Bought {qty} {ts_code} @ {price:.2f}")
            
            return {
                'success': True,
                'ts_code': ts_code,
                'qty': qty,
                'price': price,
                'side': 'buy'
            }
        
        else:  # sell
            if ts_code not in self.position:
                return {'success': False, 'reason': 'no_position'}
            
            pos = self.position[ts_code]
            if qty > pos['qty']:
                qty = pos['qty']
            
            revenue = qty * price * (1 - 0.001 - 0.0003)  # 印花税+佣金
            self.cash += revenue
            
            pos['qty'] -= qty
            pos['buy_today'] = False  # 卖出后解除T+1限制
            
            if pos['qty'] <= 0:
                del self.position[ts_code]
            
            logger.info(f"[SIM] Sold {qty} {ts_code} @ {price:.2f}")
            
            return {
                'success': True,
                'ts_code': ts_code,
                'qty': qty,
                'price': price,
                'side': 'sell'
            }
```

### 6.2 可转债交易接口

```python
# execution/cb_trader.py

class CBTrader:
    """可转债交易接口（与A股共用券商通道）"""
    
    def __init__(self):
        self.bond_position: Dict[str, Dict] = {}
        self.stock_position: Dict[str, Dict] = {}  # 转股后的正股持仓
    
    async def buy_cb(self, bond_id: str, amount_cny: float) -> Dict:
        """买入可转债"""
        # 可转债最小交易单位为10张（1000元面值）
        pass
    
    async def sell_cb(self, bond_id: str, qty: int) -> Dict:
        """卖出可转债"""
        pass
    
    async def convert_to_stock(self, bond_id: str, qty: int = None) -> Dict:
        """
        转股操作
        
        Args:
            bond_id: 可转债代码
            qty: 转股数量（不指定则全部）
        """
        if bond_id not in self.bond_position:
            return {'success': False, 'reason': 'no_position'}
        
        pos = self.bond_position[bond_id]
        convert_qty = qty or pos['qty']
        
        # 转股比例 = 面值 / 转股价
        # 获取转股比例
        conv_ratio = await self._get_conversion_ratio(bond_id)
        
        # 得到正股数量
        stock_qty = int(convert_qty * conv_ratio / 10)  # 10张=1000面值
        
        # 扣除可转债，增加正股
        pos['qty'] -= convert_qty
        stock_code = pos['linked_stock']
        
        if stock_code in self.stock_position:
            sp = self.stock_position[stock_code]
            sp['qty'] += stock_qty
        else:
            self.stock_position[stock_code] = {
                'qty': stock_qty,
                'linked_cb': bond_id
            }
        
        logger.info(f"[CONVERT] {bond_id} -> {stock_qty} {stock_code}")
        
        return {'success': True, 'bond_id': bond_id, 'stock_qty': stock_qty}
    
    async def subscribe_cb(self, bond_id: str, amount: int = 10) -> Dict:
        """申购新债"""
        # 新债申购为信用申购，无需资金
        pass
```

---

## 7. 风险管理升级

### 7.1 A股风险模型

```python
# strategy_manager/risk.py 改造

import asyncio
from typing import Dict, Optional, Tuple
from loguru import logger

class AStockRiskEngine:
    """A股风险管理"""
    
    def __init__(self):
        self.blacklist: set = set()
        self.stock_pool: set = set()
    
    async def check_safety(
        self, 
        ts_code: str, 
        price: float,
        market_cap: float = None
    ) -> Tuple[bool, str]:
        """
        安全检查
        
        Returns:
            (is_safe, reason)
        """
        # 1. ST/*ST 过滤
        if await self._is_st_stock(ts_code):
            return False, "ST stock"
        
        # 2. 涨跌停检查
        limit_status = await self._get_limit_status(ts_code)
        if limit_status == 'limit_up':
            return False, "Limit up - cannot buy"
        if limit_status == 'limit_down':
            return False, "Limit down - risk"
        
        # 3. 停牌检查
        if await self._is_halted(ts_code):
            return False, "Halted stock"
        
        # 4. 市值过滤
        if market_cap and market_cap < 5e8:  # 5亿以下不碰
            return False, "Market cap too small"
        
        # 5. 流动性过滤
        if not await self._has_liquidity(ts_code):
            return False, "Low liquidity"
        
        # 6. 退市风险检查
        if await self._has_delisting_risk(ts_code):
            return False, "Delisting risk"
        
        return True, "OK"
    
    async def calculate_position_size(
        self, 
        stock_code: str, 
        signal_score: float,
        total_capital: float,
        max_position_pct: float = 0.2
    ) -> float:
        """计算仓位"""
        # 基础仓位 = 资金 * 比例
        base_size = total_capital * max_position_pct
        
        # 根据信号强度调整
        adjusted_size = base_size * signal_score
        
        # 风险调整：如果接近涨跌停，降低仓位
        limit_status = await self._get_limit_status(stock_code)
        if limit_status == 'limit_up':
            adjusted_size *= 0.5
        
        return min(adjusted_size, total_capital * 0.3)  # 单只不超过30%
    
    async def _is_st_stock(self, ts_code: str) -> bool:
        """检查是否ST"""
        # 通过名称判断或查询数据库
        return 'ST' in ts_code or '*ST' in ts_code
    
    async def _get_limit_status(self, ts_code: str) -> str:
        """获取涨跌停状态"""
        # 实时数据获取
        return 'normal'
    
    async def _is_halted(self, ts_code: str) -> bool:
        """检查是否停牌"""
        return False
    
    async def _has_liquidity(self, ts_code: str) -> bool:
        """流动性检查"""
        # 日成交额>3000万
        return True
    
    async def _has_delisting_risk(self, ts_code: str) -> bool:
        """退市风险"""
        return False


class CBRiskEngine:
    """可转债风险管理"""
    
    def __init__(self):
        self.high_risk_cb: set = set()
    
    async def check_safety(self, bond_id: str, cb_data: Dict) -> Tuple[bool, str]:
        """可转债安全检查"""
        # 1. 强赎风险
        if cb_data['conv_value'] > 130:
            return False, "Strong redemption risk"
        
        # 2. 双高风险（价格高+溢价率高）
        if cb_data['price'] > 150 and cb_data['premium_rate'] > 50:
            return False, "Double high risk"
        
        # 3. 剩余规模过小
        if cb_data['remain_amount'] < 0.3:  # 3000万
            return False, "Low remaining scale"
        
        # 4. 临近到期
        days_to_maturity = (cb_data['maturity_date'] - datetime.now()).days
        if days_to_maturity < 90:
            return False, "Near maturity"
        
        # 5. 转股溢价率过高
        if cb_data['premium_rate'] > 100:
            return False, "High premium"
        
        return True, "OK"
    
    async def monitor_redemption(self, bond_id: str) -> bool:
        """强赎监测"""
        # 转股价值>130连续15天，发布强赎公告风险
        pass
```

### 7.2 组合优化器

```python
# strategy_manager/optimizer.py

class PortfolioOptimizer:
    """组合优化器"""
    
    def __init__(self, max_positions: int = 10):
        self.max_positions = max_positions
    
    def optimize(
        self,
        signals: Dict[str, float],
        risks: Dict[str, float],
        current_positions: Dict[str, float]
    ) -> Dict[str, float]:
        """
        组合优化
        
        Args:
            signals: {ts_code: signal_score}
            risks: {ts_code: risk_score}
            current_positions: {ts_code: current_weight}
        
        Returns:
            target_weights: {ts_code: target_weight}
        """
        # 过滤不合格标的
        candidates = {
            k: v for k, v in signals.items()
            if risks.get(k, 0) < 0.5  # 风险<0.5
        }
        
        # 按信号排序
        sorted_candidates = sorted(
            candidates.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:self.max_positions]
        
        # 简单等权分配
        n = len(sorted_candidates)
        if n == 0:
            return {}
        
        weight_per_stock = 1.0 / n
        
        return {
            ts_code: weight_per_stock * signal
            for ts_code, signal in sorted_candidates
        }
```

---

## 8. 策略运行器改造

### 8.1 A股策略运行器

```python
# strategy_manager/astock_runner.py

import asyncio
import time
import json
from typing import Dict, Optional
from loguru import logger

from data_pipeline.data_manager import DataManager
from model_core.vm import StackVM
from model_core.data_loader import AStockDataLoader
from execution.a_stock_trader import AStockTrader
from .config import AStockConfig
from .portfolio import PortfolioManager
from .risk import AStockRiskEngine

class AStockStrategyRunner:
    """A股策略运行器"""
    
    def __init__(self, config: AStockConfig = None):
        self.config = config or AStockConfig()
        self.data_mgr = DataManager()
        self.trader = AStockTrader(broker=self.config.BROKER)
        self.portfolio = PortfolioManager()
        self.risk = AStockRiskEngine()
        self.vm = StackVM()
        
        self.loader = AStockDataLoader()
        self.formula = None
        
        self.last_rebalance_time = 0
        self.is_monitoring = False
    
    async def initialize(self):
        """初始化"""
        await self.data_mgr.initialize()
        
        # 加载策略
        with open("best_astock_strategy.json", "r") as f:
            self.formula = json.load(f)
        
        logger.success(f"A-Stock Strategy Loaded: {self.formula}")
    
    async def run(self):
        """主循环"""
        logger.info("A-Stock Strategy Runner Started")
        
        while True:
            try:
                loop_start = time.time()
                
                # 每日调仓（14:50执行）
                current_minute = time.strftime("%H%M")
                if current_minute >= self.config.REBALANCE_TIME:
                    await self._daily_rebalance()
                
                # 实时监控
                if self.is_monitoring:
                    await self._monitor_positions()
                
                elapsed = time.time() - loop_start
                sleep_time = max(60, 300 - elapsed)  # 5分钟循环
                await asyncio.sleep(sleep_time)
                
            except Exception as e:
                logger.exception(f"Loop Error: {e}")
                await asyncio.sleep(60)
    
    async def _daily_rebalance(self):
        """日度调仓"""
        if time.time() - self.last_rebalance_time < 86400:  # 24小时内只调仓一次
            return
        
        logger.info("Starting Daily Rebalance...")
        
        # 加载最新数据
        end_date = time.strftime('%Y%m%d')
        start_date = self._get_date_n_days_ago(60)
        
        self.loader.load_data(start_date, end_date)
        
        # 计算信号
        signals = self.vm.execute(self.formula, self.loader.feat_tensor)
        if signals is None:
            logger.warning("Signal computation failed")
            return
        
        latest_signals = torch.sigmoid(signals[:, -1]).cpu().numpy()
        
        # 获取持仓
        positions = await self.trader.get_positions()
        
        # 目标信号排序
        sorted_indices = latest_signals.argsort()[::-1]
        
        # 选股
        target_stocks = []
        for idx in sorted_indices:
            score = latest_signals[idx]
            if score < self.config.BUY_THRESHOLD:
                break
            
            ts_code = self.loader.stock_list[idx]
            
            # 过滤已有持仓
            if ts_code in positions:
                continue
            
            # 风险检查
            is_safe, _ = await self.risk.check_safety(ts_code, 0)
            if not is_safe:
                continue
            
            target_stocks.append((ts_code, score))
            
            if len(target_stocks) >= self.config.MAX_POSITIONS:
                break
        
        # 执行调仓
        await self._execute_rebalance(target_stocks, positions)
        
        self.last_rebalance_time = time.time()
    
    async def _execute_rebalance(
        self, 
        target_stocks: list,
        current_positions: Dict
    ):
        """执行调仓"""
        # 卖出
        for ts_code in list(current_positions.keys()):
            if ts_code not in [s[0] for s in target_stocks]:
                logger.info(f"Selling {ts_code} (not in target)")
                await self.trader.sell(ts_code, percentage=1.0)
                await asyncio.sleep(0.5)
        
        # 买入
        cash = await self.trader.get_cash()
        alloc_per_stock = cash / len(target_stocks)
        
        for ts_code, score in target_stocks:
            logger.info(f"Buying {ts_code} (score={score:.2f})")
            await self.trader.buy(ts_code, alloc_per_stock)
            await asyncio.sleep(0.5)
    
    async def _monitor_positions(self):
        """持仓监控"""
        positions = await self.trader.get_positions()
        
        for ts_code, pos in positions.items():
            # 检查止损
            current_price = await self._get_price(ts_code)
            pnl = (current_price - pos['avg_price']) / pos['avg_price']
            
            if pnl < self.config.STOP_LOSS_PCT:
                logger.warning(f"Stop Loss: {ts_code} PnL={pnl:.2%}")
                await self.trader.sell(ts_code, percentage=1.0)
    
    async def _get_price(self, ts_code: str) -> float:
        """获取价格"""
        # 实时行情获取
        return 0.0
```

### 8.2 可转债策略运行器

```python
# strategy_manager/cb_runner.py

class CBStrategyRunner:
    """可转债策略运行器（与A股类似，但逻辑不同）"""
    
    async def _daily_rebalance(self):
        """可转债调仓逻辑"""
        # 1. 筛选双低因子排序
        # 2. 排除强赎风险
        # 3. 排除临期债券
        # 4. 分配仓位
        pass
    
    async def _monitor_cb_positions(self):
        """可转债持仓监控"""
        # 1. 强赎监测
        # 2. 转股价值监测
        # 3. 下修转股价预期监测
        pass
```

---

## 9. 配置系统

### 9.1 A股配置

```python
# strategy_manager/config.py 新增

@dataclass
class AStockConfig:
    """A股策略配置"""
    BROKER: str = 'simulate'  # huatai/guotai/simulate
    REBALANCE_TIME: str = '1450'  # 14:50调仓
    MAX_POSITIONS: int = 10
    BUY_THRESHOLD: float = 0.75
    STOP_LOSS_PCT: float = -0.08  # -8%止损
    TAKE_PROFIT_PCT: float = 0.15  # 15%止盈
    
    # 仓位
    MAX_POSITION_PCT: float = 0.2  # 单只最大20%
    MIN_BUY_AMOUNT: float = 10000  # 最小买入1万


@dataclass  
class CBConfig:
    """可转债策略配置"""
    MAX_POSITIONS: int = 15
    BUY_THRESHOLD: float = 0.7
    DOUBLE_LOW_THRESHOLD: float = 120  # 双低<120才买
    MIN_CONV_VALUE: float = 90  # 转股价值>90
    MAX_PREMIUM: float = 50  # 溢价率<50%
    MIN_REMAIN: float = 1.0  # 剩余规模>1亿
    STOP_LOSS_PCT: float = -0.05  # -5%止损
    TAKE_PROFIT_PCT: float = 0.10  # 10%止盈
```

### 9.2 ModelConfig 调整

```python
# model_core/config.py

class ModelConfig:
    # 原有配置...
    
    # A股特定
    BATCH_SIZE_ASTOCK = 4096  # A股数量多
    MAX_FORMULA_LEN_ASTOCK = 15  # A股因子可能更复杂
    
    # 可转债特定
    BATCH_SIZE_CB = 512  # 可转债数量有限
    MAX_FORMULA_LEN_CB = 12
    
    # 数据过滤
    MIN_MARKET_CAP_ASTOCK = 5e8  # 5亿市值
    MIN_REMAIN_CB = 1e7  # 1亿剩余规模
```

---

## 10. 实施路线图

### Phase 1: 数据层改造（2-3周）

| 任务 | 工期 | 负责人 |
|------|------|--------|
| Tushare/集思录数据接口开发 | 1周 | - |
| 数据库表结构设计与创建 | 2天 | - |
| 历史数据回填脚本 | 1周 | - |
| 数据验证与清洗 | 3天 | - |

### Phase 2: 因子与模型改造（2-3周）

| 任务 | 工期 | 负责人 |
|------|------|--------|
| A股因子库开发 | 1周 | - |
| 可转债因子库开发 | 1周 | - |
| 特征工程接口统一 | 3天 | - |
| 模型超参数调整 | 1周 | - |

### Phase 3: 回测系统改造（1-2周）

| 任务 | 工期 | 负责人 |
|------|------|--------|
| A股回测引擎（T+1/涨跌停） | 1周 | - |
| 可转债回测引擎 | 5天 | - |
| 回测验证与对比 | 3天 | - |

### Phase 4: 执行层开发（2-3周）

| 任务 | 工期 | 负责人 |
|------|------|--------|
| A股模拟交易接口 | 1周 | - |
| 可转债交易接口 | 1周 | - |
| 券商场商API对接 | 2周 | - |

### Phase 5: 实盘测试（2-4周）

| 任务 | 工期 | 负责人 |
|------|------|--------|
| 模拟盘运行验证 | 1周 | - |
| 小资金实盘测试 | 1-2周 | - |
| 策略优化迭代 | 1-2周 | - |

---

## 11. 风险与注意事项

### 11.1 技术风险

| 风险 | 影响 | 应对措施 |
|------|------|----------|
| 数据API限流 | 无法获取实时数据 | 多数据源备份+本地缓存 |
| 回测过拟合 | 实盘亏损 | 增加样本外测试+Walk-forward |
| 交易滑点 | 实际收益低于回测 | 加入滑点模型 |
| T+1规则违反 | 实盘无法成交 | 严格规则检查 |

### 11.2 市场风险

| 风险 | 影响 | 应对措施 |
|------|------|----------|
| 流动性风险 | 无法买入/卖出 | 流动性阈值过滤 |
| 政策风险 | 规则变化 | 持续监控政策 |
| 强赎风险 | 可转债强制赎回 | 实时监测+提前卖出 |

### 11.3 合规风险

| 风险 | 影响 | 应对措施 |
|------|------|----------|
| 账户权限 | 无法交易 | 提前申请权限 |
| 交易限制 | 触发风控 | 设置交易限制参数 |

---

## 12. 验收标准

### 12.1 功能验收

- [ ] 数据覆盖 A 股全市场（>5000 只）
- [ ] 可转债数据完整（>500 只）
- [ ] 因子计算无错误
- [ ] 回测与实盘结果差异 < 5%

### 12.2 性能验收

- [ ] 数据加载 < 30秒（60天数据）
- [ ] 单次因子计算 < 10秒
- [ ] 回测执行 < 2分钟
- [ ] 实盘延迟 < 5秒

### 12.3 稳定性验收

- [ ] 连续运行 7 天无崩溃
- [ ] 数据API异常自动恢复
- [ ] 交易指令 100% 执行成功

---

## 13. 附录

### 13.1 关键文件清单

| 文件路径 | 修改类型 | 说明 |
|----------|----------|------|
| `data_pipeline/providers/tushare.py` | 新增 | Tushare数据接口 |
| `data_pipeline/providers/jisilu.py` | 新增 | 集思录数据接口 |
| `data_pipeline/db_manager.py` | 修改 | 数据库表结构调整 |
| `data_pipeline/schema.py` | 新增 | 数据模型定义 |
| `model_core/data_loader.py` | 重构 | 支持A股/可转债 |
| `model_core/factors.py` | 重构 | 新增因子库 |
| `model_core/backtest.py` | 重构 | T+1/涨跌停逻辑 |
| `execution/a_stock_trader.py` | 新增 | A股交易接口 |
| `execution/cb_trader.py` | 新增 | 可转债交易接口 |
| `strategy_manager/risk.py` | 重构 | A股/可转债风控 |
| `strategy_manager/astock_runner.py` | 新增 | A股策略运行器 |
| `strategy_manager/cb_runner.py` | 新增 | 可转债策略运行器 |
| `strategy_manager/portfolio.py` | 修改 | 支持A股持仓 |
| `strategy_manager/config.py` | 新增 | A股/可转债配置 |

### 13.2 依赖更新

```txt
# requirements.txt 新增
tushare>=1.3.0          # A股数据
akshare>=1.4.0          # 可转债数据
jqdatasdk>=1.8.0        # 聚宽数据（可选）
```

### 13.3 参考文献

1. Tushare Pro 文档: https://tushare.pro/document/1
2. 集思录可转债数据: https://www.jisilu.cn/data/cb/
3. A 股交易规则: 证券交易所官网
4. 可转债投资指南: 集思录学堂
