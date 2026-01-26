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
    
    @staticmethod
    def compute_features(raw_data: dict) -> torch.Tensor:
        """
        从原始数据中提取模型输入特征
        
        Args:
            raw_data: dict of tensors, 每个 key 对应一个 BASIC_FACTOR 的内部名称
                      每个 tensor 的形状为 [Time, Assets]
        
        Returns:
            feat_tensor: [Time, Assets, Features] 用于 StackVM 执行的输入张量
        """
        features = []
        
        for feat_name in ModelConfig.INPUT_FEATURES:
            if feat_name not in raw_data:
                raise KeyError(f"Feature '{feat_name}' not found in raw_data. "
                               f"Available: {list(raw_data.keys())}")
            
            feat = raw_data[feat_name]  # [Time, Assets]
            
            # 对每个特征做一个基础的标准化 (Robust Normalization)
            # 使用中位数和 MAD，避免异常值影响
            feat_norm = FeatureEngineer._robust_normalize(feat)
            
            features.append(feat_norm)
        
        # Stack 成 [Time, Assets, Features]
        feat_tensor = torch.stack(features, dim=2)
        
        return feat_tensor
    
    @staticmethod
    def _robust_normalize(t: torch.Tensor, window: int = 60) -> torch.Tensor:
        """
        稳健标准化 (Rolling Robust Normalization)
        
        使用 过去N天 的滚动均值和标准差进行标准化，严格消除未来函数。
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
        padded = torch.cat([padding, t], dim=0) # [T+window-1, N]
        
        unfolded = padded.unfold(0, window, 1) # [T, N, window]
        
        # 计算滚动统计量
        roll_mean = unfolded.mean(dim=-1) # [T, N]
        roll_std = unfolded.std(dim=-1) + 1e-9 # [T, N]
        
        # Z-Score
        norm = (t - roll_mean) / roll_std
        
        # 前 window 天由于 padding 导致数据不准，设为 0
        norm[:window] = 0.0
        
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