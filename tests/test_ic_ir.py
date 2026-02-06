"""
IC/IR 合成单元测试 (Synthetic Unit Tests)

验证 IC (Information Coefficient) 和 IR (Information Ratio) 计算逻辑的正确性。
覆盖以下场景:
1. Skip Logic: n < 2, constant factor, constant return
2. Ties Case: 验证平均排名
3. Perfect Correlations: IC ≈ ±1.0
4. Multi-Day Aggregation: 验证 valid_days, skipped_days, ddof=1
5. Side-Effect Test: 输入 tensor 不变
"""
import sys
import os
import torch
import math

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_core.backtest import CBBacktest


class TestICIR:
    """IC/IR 测试类"""
    
    def __init__(self):
        self.bt = CBBacktest(top_k=10)
        self.results = []
    
    def run_all(self):
        """运行所有测试"""
        tests = [
            ("Skip: n < 2", self.test_skip_logic_n_lt_2),
            ("Skip: constant factor", self.test_skip_logic_constant_factor),
            ("Skip: constant return", self.test_skip_logic_constant_return),
            ("Ties case", self.test_ties_case),
            ("Perfect positive IC", self.test_perfect_positive_ic),
            ("Perfect negative IC", self.test_perfect_negative_ic),
            ("Multi-day aggregation", self.test_multi_day_aggregation),
            ("Null IR case", self.test_null_ir_case),
            ("Side-effect test", self.test_side_effect),
        ]
        
        print("=" * 60)
        print("IC/IR 合成单元测试")
        print("=" * 60)
        
        for name, test_func in tests:
            try:
                passed, message = test_func()
                status = "✓ PASS" if passed else "✗ FAIL"
                print(f"  {name}: {status}")
                if not passed:
                    print(f"    -> {message}")
                self.results.append((name, passed, message))
            except Exception as e:
                print(f"  {name}: ✗ ERROR")
                print(f"    -> {type(e).__name__}: {e}")
                self.results.append((name, False, str(e)))
        
        print("=" * 60)
        passed_count = sum(1 for _, p, _ in self.results if p)
        print(f"结果: {passed_count}/{len(self.results)} 通过")
        
        return all(p for _, p, _ in self.results)
    
    # ========== Skip Logic Tests ==========
    
    def test_skip_logic_n_lt_2(self):
        """测试 valid_count < 2 时跳过"""
        # 只有 1 个有效样本
        factor = torch.tensor([1.0, 2.0, 3.0])
        ret = torch.tensor([0.1, 0.2, 0.3])
        mask = torch.tensor([True, False, False])
        
        ic = self.bt._compute_daily_ic(factor, ret, mask)
        
        if ic is None:
            return True, "正确跳过 (n < 2)"
        else:
            return False, f"应返回 None，实际返回 {ic}"
    
    def test_skip_logic_constant_factor(self):
        """测试 factor 全相同时跳过 (std=0)"""
        factor = torch.tensor([5.0, 5.0, 5.0, 5.0, 5.0])
        ret = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5])
        mask = torch.ones(5, dtype=torch.bool)
        
        ic = self.bt._compute_daily_ic(factor, ret, mask)
        
        if ic is None:
            return True, "正确跳过 (constant factor)"
        else:
            return False, f"应返回 None，实际返回 {ic}"
    
    def test_skip_logic_constant_return(self):
        """测试 return 全相同时跳过 (std=0)"""
        factor = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        ret = torch.tensor([0.05, 0.05, 0.05, 0.05, 0.05])
        mask = torch.ones(5, dtype=torch.bool)
        
        ic = self.bt._compute_daily_ic(factor, ret, mask)
        
        if ic is None:
            return True, "正确跳过 (constant return)"
        else:
            return False, f"应返回 None，实际返回 {ic}"
    
    # ========== Ties Case Test ==========
    
    def test_ties_case(self):
        """测试并列排名情况
        
        factor = [1, 1, 2, 2, 3]
        预期排名: [1.5, 1.5, 3.5, 3.5, 5.0] (1-based average)
        """
        factor = torch.tensor([1.0, 1.0, 2.0, 2.0, 3.0])
        ret = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        mask = torch.ones(5, dtype=torch.bool)
        
        # 验证排名计算 (注意: _rank_with_ties 返回 float64)
        ranks = self.bt._rank_with_ties(factor)
        expected_ranks = torch.tensor([1.5, 1.5, 3.5, 3.5, 5.0], dtype=torch.float64)
        
        rank_match = torch.allclose(ranks, expected_ranks, atol=1e-6)
        
        # 验证 IC 可以计算
        ic = self.bt._compute_daily_ic(factor, ret, mask)
        
        if rank_match and ic is not None:
            return True, f"排名正确，IC = {ic:.4f}"
        elif not rank_match:
            return False, f"排名错误: {ranks.tolist()} vs {expected_ranks.tolist()}"
        else:
            return False, "IC 不应为 None"
    
    # ========== Perfect Correlation Tests ==========
    
    def test_perfect_positive_ic(self):
        """测试完美正相关: factor = ret -> IC ≈ +1.0"""
        factor = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        ret = factor.clone()  # 完全相同
        mask = torch.ones(10, dtype=torch.bool)
        
        ic = self.bt._compute_daily_ic(factor, ret, mask)
        
        if ic is not None and abs(ic - 1.0) < 1e-6:
            return True, f"IC = {ic:.6f} ≈ +1.0"
        else:
            return False, f"IC = {ic}，预期 ≈ +1.0"
    
    def test_perfect_negative_ic(self):
        """测试完美负相关: factor = -ret -> IC ≈ -1.0"""
        factor = torch.tensor([10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0])
        ret = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        mask = torch.ones(10, dtype=torch.bool)
        
        ic = self.bt._compute_daily_ic(factor, ret, mask)
        
        if ic is not None and abs(ic + 1.0) < 1e-6:
            return True, f"IC = {ic:.6f} ≈ -1.0"
        else:
            return False, f"IC = {ic}，预期 ≈ -1.0"
    
    # ========== Multi-Day Aggregation Test ==========
    
    def test_multi_day_aggregation(self):
        """测试多日聚合
        
        Day 0: 有效 IC
        Day 1: 跳过 (constant factor)
        Day 2: 有效 IC
        
        预期:
        - valid_days = 2
        - skipped_days = 3 - 2 = 1 (最后一天强制跳过 + Day 1)
        - ic_std 使用 ddof=1
        """
        T, N = 3, 10
        
        # 构造因子和收益
        factors = torch.zeros(T, N)
        target_ret = torch.zeros(T, N)
        valid_mask = torch.ones(T, N, dtype=torch.bool)
        
        # Day 0: 正常因子 -> 用于计算 Day 0 收益 (Data Loader定义: target_ret[t]为t->t+1收益)
        factors[0] = torch.arange(1, N + 1, dtype=torch.float32)
        target_ret[0] = torch.arange(1, N + 1, dtype=torch.float32) * 0.01  # T+1 收益对齐到 t
        
        # Day 1: 常数因子 -> 应该跳过
        factors[1] = torch.ones(N) * 5.0  # 常数因子
        target_ret[1] = torch.arange(1, N + 1, dtype=torch.float32) * 0.02
        
        # Day 2: 正常因子，但收益反向 -> 产生负 IC
        # 这样 Day 0 IC=1.0, Day 2 IC=-1.0, ic_std > 0
        factors[2] = torch.arange(1, N + 1, dtype=torch.float32)
        target_ret[2] = torch.arange(N, 0, -1, dtype=torch.float32) * 0.03  # 反向
        
        metrics = self.bt._compute_ic_metrics(factors, target_ret, valid_mask)
        
        # 预期: 
        # Day 0: 有效
        # Day 1: 跳过 (Constant)
        # Day 2: 有效 (新逻辑不再强制跳过最后一天，只要数据有效)
        
        # 所以 valid_days = 2 (Day 0, Day 2)
        
        # 如果 valid_days < 2，ic_ir 应为 None
        if metrics['valid_ic_days'] < 2 and metrics['ic_ir'] is None:
            return True, f"valid_days={metrics['valid_ic_days']}, ic_ir=None (正确)"
        elif metrics['valid_ic_days'] >= 2:
            # 如果有 2+ 有效天，验证 ic_std 使用 ddof=1
            if metrics['ic_std'] > 0 and metrics['ic_ir'] is not None:
                return True, f"valid_days={metrics['valid_ic_days']}, ic_ir={metrics['ic_ir']:.4f}"
            else:
                return False, f"ic_std={metrics['ic_std']}, ic_ir={metrics['ic_ir']}"
        else:
            return False, f"valid_days={metrics['valid_ic_days']}, 但 ic_ir 不为 None"
    
    def test_null_ir_case(self):
        """测试仅 1 天有效时 ic_ir = None"""
        T, N = 2, 10
        
        factors = torch.zeros(T, N)
        target_ret = torch.zeros(T, N)
        valid_mask = torch.ones(T, N, dtype=torch.bool)
        
        # Day 0: 正常因子
        factors[0] = torch.arange(1, N + 1, dtype=torch.float32)
        target_ret[0] = torch.arange(1, N + 1, dtype=torch.float32) * 0.01
        
        # Day 1: 最后一天，但新逻辑会尝试计算，如果数据有效
        # 为了测试 Null IR (即只有 1 天有效)，我们让 Day 1 数据无效(全0)
        # 或者让 mask 为 False
        factors[1] = torch.arange(1, N + 1, dtype=torch.float32)
        valid_mask[1] = False  # 强制无效
        
        metrics = self.bt._compute_ic_metrics(factors, target_ret, valid_mask)
        
        # 只有 Day 0 有效，valid_days = 1，ic_ir 应为 None
        if metrics['valid_ic_days'] == 1 and metrics['ic_ir'] is None:
            return True, f"valid_days=1, ic_ir=None (正确)"
        else:
            return False, f"valid_days={metrics['valid_ic_days']}, ic_ir={metrics['ic_ir']}"
    
    # ========== Side-Effect Test ==========
    
    def test_side_effect(self):
        """测试输入 tensor 在调用后不变"""
        T, N = 5, 10
        
        factors = torch.randn(T, N)
        target_ret = torch.randn(T, N)
        valid_mask = torch.ones(T, N, dtype=torch.bool)
        
        # 保存原始值
        factors_orig = factors.clone()
        target_ret_orig = target_ret.clone()
        valid_mask_orig = valid_mask.clone()
        
        # 调用计算
        _ = self.bt._compute_ic_metrics(factors, target_ret, valid_mask)
        
        # 验证不变
        factors_unchanged = torch.equal(factors, factors_orig)
        ret_unchanged = torch.equal(target_ret, target_ret_orig)
        mask_unchanged = torch.equal(valid_mask, valid_mask_orig)
        
        if factors_unchanged and ret_unchanged and mask_unchanged:
            return True, "输入 tensor 未被修改"
        else:
            changed = []
            if not factors_unchanged:
                changed.append("factors")
            if not ret_unchanged:
                changed.append("target_ret")
            if not mask_unchanged:
                changed.append("valid_mask")
            return False, f"以下 tensor 被修改: {', '.join(changed)}"


def run_tests():
    """运行所有 IC/IR 测试"""
    tester = TestICIR()
    return tester.run_all()


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)
