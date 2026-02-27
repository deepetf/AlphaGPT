"""
特征工程模块 (Feature Engineering)

使用配置驱动的方式构建特征，支持灵活扩展。
"""
import torch
from .config import ModelConfig


class FeatureEngineer:
    """
    动态特征工程器
    
    根据 ModelConfig.INPUT_FEATURES 自动构建输入张量
    """
    INPUT_DIM = ModelConfig.INPUT_DIM

    # 按需派生特征依赖的原始特征定义。
    # key: derived feature name
    # value: tuple of required raw feature names
    DERIVED_FEATURE_SOURCES = {
        "LOG_MONEYNESS": ("CLOSE_STK", "CONV_PRICE"),
    }
    
    @staticmethod
    def compute_features(raw_data: dict, warmup_rows: int = 0) -> torch.Tensor:
        """
        从原始数据中提取模型输入特征
        
        Args:
            raw_data: dict of tensors, 每个 key 对应一个 BASIC_FACTOR 的内部名称
                      每个 tensor 的形状为 [Time, Assets]
            warmup_rows: 预热段行数。当 > 0 时，表示数据开头有真实的预热数据，
                         _robust_normalize 将减少或跳过前 window 行的零化操作。
        
        Returns:
            feat_tensor: [Time, Assets, Features] 用于 StackVM 执行的输入张量
        """
        features = []
        feature_cache = {}

        def get_feature_tensor(feat_name: str) -> torch.Tensor:
            if feat_name in feature_cache:
                return feature_cache[feat_name]

            if feat_name in raw_data:
                tensor = raw_data[feat_name]
            elif feat_name == "LOG_MONEYNESS":
                tensor = FeatureEngineer._compute_log_moneyness(get_feature_tensor)
            else:
                raise KeyError(
                    f"Feature '{feat_name}' not found in raw_data or derived registry. "
                    f"Available raw: {list(raw_data.keys())}, "
                    f"available derived: {list(FeatureEngineer.DERIVED_FEATURE_SOURCES.keys())}"
                )

            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"Feature '{feat_name}' must be torch.Tensor, got {type(tensor)}")
            if tensor.dim() != 2:
                raise ValueError(
                    f"Feature '{feat_name}' must be 2D [Time, Assets], got shape={tuple(tensor.shape)}"
                )

            feature_cache[feat_name] = tensor
            return tensor

        for feat_name in ModelConfig.INPUT_FEATURES:
            feat = get_feature_tensor(feat_name)

            # 对每个特征做稳健标准化 (Rolling Robust Normalization)
            feat_norm = FeatureEngineer._robust_normalize(feat, warmup_rows=warmup_rows)
            features.append(feat_norm)
        
        # Stack 成 [Time, Assets, Features]
        feat_tensor = torch.stack(features, dim=2)
        
        return feat_tensor

    @staticmethod
    def _compute_log_moneyness(get_feature_tensor) -> torch.Tensor:
        """
        LOG_MONEYNESS = log(S / K)
        - S: CLOSE_STK (正股收盘价)
        - K: CONV_PRICE (转股价)
        """
        stock_close = get_feature_tensor("CLOSE_STK")
        conv_price = get_feature_tensor("CONV_PRICE")

        valid = (stock_close > 0) & (conv_price > 0)
        ratio = torch.where(
            valid,
            stock_close / (conv_price + 1e-9),
            torch.ones_like(stock_close),
        )
        out = torch.where(valid, torch.log(torch.clamp(ratio, min=1e-12)), torch.zeros_like(stock_close))
        return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

    @classmethod
    def get_optional_raw_feature_names(cls):
        """
        返回派生特征依赖的原始特征集合。
        这些原始特征不一定总是被使用（取决于 INPUT_FEATURES 是否启用对应派生特征）。
        """
        names = set()
        for src_names in cls.DERIVED_FEATURE_SOURCES.values():
            for n in src_names:
                names.add(n)
        return names
    
    @staticmethod
    def _robust_normalize(t: torch.Tensor, window: int = 60,
                          warmup_rows: int = 0) -> torch.Tensor:
        """
        稳健标准化 (Rolling Robust Normalization)
        
        使用 过去N天 的滚动均值和标准差进行标准化，严格消除未来函数。
        
        Args:
            t: [Time, Assets] 原始特征张量
            window: 滚动窗口大小（天数）
            warmup_rows: 预热段行数。当数据包含预热段时，前 warmup_rows 行
                         是真实数据而非人造填充，因此可以减少或跳过零化操作。
                         - warmup_rows=0: 旧行为，前 window 行全部置 0
                         - warmup_rows >= window: 不零化任何行（滚动窗口完全由真实数据填充）
                         - 0 < warmup_rows < window: 零化前 (window - warmup_rows) 行
        """
        if t.dim() != 2:
            return t
        
        T, N = t.shape
        if T < window:
            return torch.zeros_like(t)
        
        # 构造滚动窗口 [T, N, window]
        # 注意：unfold 的行为是取未来 window 个，所以为了取"过去"，需要 padding
        # 我们需要 t 时刻看到的是 t-window+1 到 t
        
        # Padding: 前面补 window-1 行，确保第0个时间点有 window 个数据(包含自己)
        # 用首行重复填充
        padding = t[0].unsqueeze(0).repeat(window-1, 1)
        padded = torch.cat([padding, t], dim=0)  # [T+window-1, N]
        
        unfolded = padded.unfold(0, window, 1)  # [T, N, window]
        
        # 计算滚动统计量
        roll_mean = unfolded.mean(dim=-1)  # [T, N]
        roll_std = unfolded.std(dim=-1) + 1e-9  # [T, N]
        
        # Z-Score
        norm = (t - roll_mean) / roll_std
        
        # 零化不可靠区域：
        # - 无预热(warmup_rows=0)：前 window 行完全依赖 padding，全部置 0
        # - 有预热(warmup_rows>0)：真实数据覆盖了部分窗口，减少零化范围
        # - 预热充足(warmup_rows>=window)：滚动窗口完全由真实数据填充，无需零化
        zero_rows = max(0, window - warmup_rows)
        if zero_rows > 0:
            norm[:zero_rows] = 0.0
        
        # Clip 极值
        return torch.clamp(norm, -5.0, 5.0)


# ============================================================
# 衍生特征计算器 (可选，用户可扩展)
# ============================================================
class DerivedFeatures:
    """
    计算衍生特征 (如收益率、动量等)
    
    这些特征可以后续加入 INPUT_FEATURES 配置中
    """
    
    @staticmethod
    def compute_return(close: torch.Tensor, lag: int = 1) -> torch.Tensor:
        """计算 lag 日收益率"""
        prev = torch.roll(close, lag, dims=0)
        ret = (close - prev) / (prev + 1e-9)
        ret[:lag] = 0  # 前 lag 天无效
        return ret
    
    @staticmethod
    def compute_momentum(close: torch.Tensor, window: int = 5) -> torch.Tensor:
        """计算 window 日动量 (当前价 / window日前价 - 1)"""
        prev = torch.roll(close, window, dims=0)
        mom = (close / (prev + 1e-9)) - 1.0
        mom[:window] = 0
        return mom
    
    @staticmethod
    def compute_volatility(close: torch.Tensor, window: int = 20) -> torch.Tensor:
        """计算 window 日波动率"""
        if close.shape[0] < window:
            return torch.zeros_like(close)
        
        # 先算收益率
        ret = DerivedFeatures.compute_return(close, lag=1)
        
        # 滚动标准差
        T, N = ret.shape
        padded = torch.cat([ret[:window-1], ret], dim=0)
        unfolded = padded.unfold(0, window, 1)  # [T, N, window]
        vol = unfolded.std(dim=-1)
        
        return vol
