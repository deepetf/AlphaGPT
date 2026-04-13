"""
AlphaGPT 璁粌寮曟搸 (鍙浆鍊虹増)

浣跨敤 Policy Gradient 璁粌 Transformer 妯″瀷鐢熸垚 Alpha 鍥犲瓙鍏紡銆?
V2: 闆嗘垚绋冲仴鎬ц瘎浼?(鍒嗘楠岃瘉銆佹粴鍔ㄧǔ瀹氭€с€佹渶澶у洖鎾ゃ€佸彲浜ゆ槗鎬х害鏉?
"""
import torch
from torch.distributions import Categorical
from tqdm import tqdm
import json
import os
import re
import time
import argparse
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from collections import Counter, OrderedDict, deque

from .config import ModelConfig, RobustConfig

from .data_loader import CBDataLoader
from .alphagpt import AlphaGPT
from .vm import StackVM
from .backtest import CBBacktest
from .ops_registry import OpsRegistry
from .formula_validator import (
    DISCRETE_OPS,
    FORBIDDEN_TERMINAL_OPS,
    HARD_FORBIDDEN_SEQUENCES,
    MAX_TERMINAL_DISCRETE,
    MAX_SIGN_LOG_DISTANCE,
)
from .formula_simplifier import formula_to_canonical_key, simplify_formula
from workflow.run_manifest import prepare_training_run, update_training_manifest


# 鍏ㄥ眬鍙橀噺锛岀敤浜庡瓙杩涚▼鍏变韩鍙鏁版嵁 (閬垮厤 Pickling 寮€閿€)
_global_vm = None
_global_bt = None
_global_feat = None
_global_ret = None
_global_mask = None
_global_cs_mask = None
_global_split_idx = None  # 璁粌/楠岃瘉鍒囧垎绱㈠紩
# 姝㈢泩鎵€闇€浠锋牸鏁版嵁
_global_open = None
_global_high = None
_global_prev_close = None

def _init_worker(feat_tensor, target_ret, valid_mask, cs_mask, split_idx, 
                 open_prices=None, high_prices=None, prev_close=None, config_path=None):
    """瀛愯繘绋嬪垵濮嬪寲鍑芥暟"""
    # 鍏抽敭淇: 瀛愯繘绋嬮渶瑕侀噸鏂板姞杞藉姩鎬侀厤缃紝鍚﹀垯 INPUT_FEATURES 涓虹┖瀵艰嚧鏍￠獙澶辫触
    if config_path:
        try:
            from .config_loader import load_config
            load_config(config_path)
        except Exception as e:
            print(f"[Worker] Failed to load config from {config_path}: {e}")

    global _global_vm, _global_bt, _global_feat, _global_ret, _global_mask, _global_cs_mask, _global_split_idx
    global _global_open, _global_high, _global_prev_close
    
    # 鍏抽敭浼樺寲: 寮哄埗鍗曠嚎绋嬭繍琛岋紝闃叉澶氳繘绋?CPU 绔炰簤 (Oversubscription)
    torch.set_num_threads(1)
    
    _global_vm = StackVM()
    _global_bt = CBBacktest(top_k=RobustConfig.TOP_K, take_profit=RobustConfig.TAKE_PROFIT)
    
    # 灏?Tensor 绉诲姩鍒?CPU 浠ラ伩鍏嶅杩涚▼ CUDA/XPU 鍐茬獊
    _global_feat = feat_tensor.to('cpu')
    _global_ret = target_ret.to('cpu')
    _global_mask = valid_mask.to('cpu')
    _global_cs_mask = cs_mask.to('cpu')
    _global_split_idx = split_idx
    
    # 姝㈢泩浠锋牸鏁版嵁
    _global_open = open_prices.to('cpu') if open_prices is not None else None
    _global_high = high_prices.to('cpu') if high_prices is not None else None
    _global_prev_close = prev_close.to('cpu') if prev_close is not None else None

def _worker_eval(formula):
    """
    瀛愯繘绋嬫墽琛屽嚱鏁?(V2: 绋冲仴鎬ц瘎浼?
    
    浣跨敤 evaluate_robust 鑾峰彇澶氱淮鎸囨爣锛屽苟璁＄畻缁煎悎濂栧姳銆?
    """
    global _global_vm, _global_bt, _global_feat, _global_ret, _global_mask, _global_cs_mask, _global_split_idx
    global _global_open, _global_high, _global_prev_close
    
    try:
        # 0. 鍏紡缁撴瀯楠岃瘉 (鍦ㄦ槀璐电殑鍥炴祴涔嬪墠杩涜)
        from .formula_validator import validate_formula
        is_valid, structural_penalty, reason = validate_formula(formula)
        
        if not is_valid:
            return RobustConfig.PENALTY_STRUCT, None, "STRUCT_INVALID", reason
        
        # 1. 鎵ц鍏紡
        try:
            res = _global_vm.execute(formula, _global_feat, cs_mask=_global_cs_mask)
            if res is None:
                return RobustConfig.PENALTY_EXEC, None, "EXEC_NONE", "Returned None"
        except Exception as e:
            return RobustConfig.PENALTY_EXEC, None, "EXEC_ERR", type(e).__name__
        
        # 2. 妫€鏌ュ洜瀛愭柟宸?
        var_threshold = 1e-4
        if res.std() < var_threshold:
            return RobustConfig.PENALTY_LOWVAR, None, "LOW_VARIANCE", f"std={res.std():.2e}, thr={var_threshold}"


        # 3. 绋冲仴鎬ц瘎浼?(鍚鐩堥€昏緫)
        metrics = _global_bt.evaluate_robust(
            factors=res,
            target_ret=_global_ret,
            valid_mask=_global_mask,
            split_idx=_global_split_idx,
            open_prices=_global_open,
            high_prices=_global_high,
            prev_close=_global_prev_close
        )
        
        # 4. 纭窐姹版潯浠?(Hard Filters) - [V3.5] 澧炲姞绾挎€ф搴︿笌 Clamp
        gaps = []
        fail_status = "PASS"
        fail_reason = "OK"
        
        # 4.1 楠岃瘉闆?Sharpe 澶綆
        if metrics['sharpe_val'] < RobustConfig.MIN_SHARPE_VAL:
            # 褰掍竴鍖?Gap: (闃堝€?- 瀹為檯鍊? / max(1, abs(闃堝€?)
            sharpe_gap = (RobustConfig.MIN_SHARPE_VAL - metrics['sharpe_val']) / max(1.0, abs(RobustConfig.MIN_SHARPE_VAL))
            gaps.append(max(0.0, float(sharpe_gap)))
            if fail_status == "PASS":
                fail_status, fail_reason = "METRIC_SHARPE", f"val={metrics['sharpe_val']:.2f}"
        
        # 4.2 娲昏穬鐜囧お浣?(閫変笉鍒拌冻澶熸爣鐨?
        if metrics['active_ratio'] < RobustConfig.MIN_ACTIVE_RATIO:
            active_gap = (RobustConfig.MIN_ACTIVE_RATIO - metrics['active_ratio']) / RobustConfig.MIN_ACTIVE_RATIO
            gaps.append(max(0.0, float(active_gap)))
            if fail_status == "PASS":
                fail_status, fail_reason = "METRIC_ACTIVE", f"ratio={metrics['active_ratio']:.2f}"
        
        # 4.4 鏈夋晥浜ゆ槗鏃ュお灏?(缁熻涓嶅彲闈?
        min_days = RobustConfig.MIN_VALID_DAYS
        if metrics['valid_days_train'] < min_days or metrics['valid_days_val'] < min_days:
            worst_days = min(metrics['valid_days_train'], metrics['valid_days_val'])
            days_gap = (min_days - worst_days) / min_days
            gaps.append(max(0.0, float(days_gap)))
            if fail_status == "PASS":
                fail_status, fail_reason = "METRIC_DAYS", f"tr={metrics['valid_days_train']},val={metrics['valid_days_val']}"

        min_valid_day_ratio = float(RobustConfig.MIN_VALID_DAY_RATIO)
        if min_valid_day_ratio > 0 and metrics['valid_day_ratio'] < min_valid_day_ratio:
            ratio_gap = (min_valid_day_ratio - metrics['valid_day_ratio']) / max(min_valid_day_ratio, 1e-9)
            gaps.append(max(0.0, float(ratio_gap)))
            if fail_status == "PASS":
                fail_status, fail_reason = "METRIC_VALID_RATIO", f"ratio={metrics['valid_day_ratio']:.3f}"

        has_metric_gap_fail = len(gaps) > 0
        avg_gap = (sum(gaps) / len(gaps)) if has_metric_gap_fail else 0.0
        metric_fail_mode = RobustConfig.METRIC_FAIL_REWARD_MODE
        if has_metric_gap_fail and metric_fail_mode == "hard":
            # 旧版逻辑：hard clamp 后立即返回
            p_max = RobustConfig.PENALTY_METRIC_MAX
            p_min = RobustConfig.PENALTY_METRIC_MIN
            penalty = p_max - (p_max - p_min) * avg_gap
            penalty = max(float(p_min), min(float(p_max), penalty))
            return float(penalty), None, fail_status, f"{fail_reason} | gap={avg_gap:.2f}"

        
        # 5. 缁煎悎璇勫垎 (Soft Scoring)
        # 5.1 鍩虹鍒? 鍔犳潈 Sharpe
        base_score = (RobustConfig.TRAIN_WEIGHT * metrics['sharpe_train'] + 
                      RobustConfig.VAL_WEIGHT * metrics['sharpe_val'])
        
        # 5.2 绋冲畾鎬у姞鎴?(Mean - K*Std 瓒婇珮瓒婂ソ)
        stability_bonus = metrics['stability_metric'] * RobustConfig.STABILITY_W
        
        # 5.3 骞村寲鏀剁泭鐜囧鍔?(榧撳姳楂樺洖鎶ョ瓥鐣?
        ret_bonus = metrics['annualized_ret'] * RobustConfig.RET_W
        
        # 5.4 鍥炴挙鎯╃綒
        mdd_penalty = metrics['max_drawdown'] * RobustConfig.MDD_W
        
        # 5.5 闀垮害鎯╃綒
        len_penalty = len(formula) * RobustConfig.LEN_W
        
        # 5.6 鍏紡缁撴瀯鎯╃綒 (鏉ヨ嚜 validate_formula)
        # structural_penalty 鏄礋鏁版垨 0锛岀洿鎺ュ姞鍒板垎鏁颁笂
        
        # 5.7 [New] 缈昏浆杞儵缃?(Dynamic Soft Penalty)
        flip_penalty = 0.0
        is_flip = False
        detail_msg = "OK"
        if metrics['sharpe_train'] * metrics['sharpe_val'] < 0:
            is_flip = True
            # 鍔ㄦ€佹儵缃? 缃氬垎涓?Val 鐨勪簭鎹熺▼搴︽垚姝ｆ瘮: Penalty = -1 * COEF * abs(Val)
            flip_penalty = -1.0 * RobustConfig.PENALTY_FLIP_COEF * abs(metrics['sharpe_val'])
            detail_msg = f"Soft Flip ({flip_penalty:.2f}, val={metrics['sharpe_val']:.2f})"

        # 5.8 鏈€缁堝垎鏁?
        final_score = (base_score + stability_bonus) * RobustConfig.SCALE + ret_bonus - mdd_penalty - len_penalty + structural_penalty + flip_penalty
        
        # 杩斿洖鍒嗘暟鍜岃缁嗕俊鎭?
        if has_metric_gap_fail:
            if metric_fail_mode == "soft":
                fail_reward = final_score - RobustConfig.METRIC_GAP_W * avg_gap
                fail_reward = min(float(RobustConfig.METRIC_FAIL_REWARD_CAP), float(fail_reward))
                fail_reward = max(float(RobustConfig.METRIC_FAIL_REWARD_FLOOR), float(fail_reward))
                return float(fail_reward), None, fail_status, f"{fail_reason} | gap={avg_gap:.2f} | mode=soft"
            # 鏈煡 mode 鍏滃簳涓?hard
            p_max = RobustConfig.PENALTY_METRIC_MAX
            p_min = RobustConfig.PENALTY_METRIC_MIN
            penalty = p_max - (p_max - p_min) * avg_gap
            penalty = max(float(p_min), min(float(p_max), penalty))
            return float(penalty), None, fail_status, f"{fail_reason} | gap={avg_gap:.2f} | mode=hard_fallback"

        if is_flip:
            # [Candidate Isolation] 鏈夊垎鏁?RL鍙涔?锛屼絾info=None(涓嶄綔涓篕ing鍊欓€?
            return final_score, None, "METRIC_FLIP", detail_msg
        else:
            return final_score, (final_score, metrics['annualized_ret'], metrics['sharpe_all'], formula, metrics), "PASS", "OK"
    
    except (ImportError, NameError, AttributeError, SyntaxError) as e:
        # 绯荤粺绾ч敊璇細鐩存帴鎶涘嚭锛屼腑鏂缁冿紝鏂逛究 Debug (濡傚垰鎵嶇殑 ImportError)
        import traceback
        traceback.print_exc()
        raise e  # 璁╀富杩涚▼鎰熺煡鍒?Worker 鎸備簡

    except Exception:
        # 杩愯鏃堕敊璇?(濡傞櫎闆躲€丯aN銆佺煩闃靛昂瀵镐笉鍖归厤)锛氳涓哄叕寮忔棤鏁堬紝缁欐渶浣庡垎
        # 濡傛灉闇€瑕佽皟璇曪紝鍙互鎵撳紑涓嬮潰鐨勬敞閲?
        # import traceback
        # traceback.print_exc()
        return RobustConfig.PENALTY_EXEC, None, "EXEC_ERR", "RuntimeError"



class AlphaEngine:
    def __init__(self, data_start_date=None, run_context=None):
        print("Initializing AlphaEngine...")
        # 鎵撳嵃閰嶇疆鏉ユ簮
        config_source = getattr(RobustConfig, '_config_path', 'default_config.yaml')
        print(f"Config source: {config_source}")
        print(f"Using Device: {ModelConfig.DEVICE}")
        print(f"Take Profit: {RobustConfig.TAKE_PROFIT}")
        self.run_context = run_context or {}
        if self.run_context:
            print(f"Run ID: {self.run_context.get('run_id')}")
            print(f"Artifacts Dir: {self.run_context.get('run_dir')}")
        
        # 1. 鍒濆鍖栧苟鍔犺浇鏁版嵁
        # 纭繚 engine 鎷ユ湁鍞竴鐨勬暟鎹姞杞藉櫒瀹炰緥
        self.loader = CBDataLoader()
        effective_data_start_date = data_start_date or "2022-08-01"
        print(f"Data start date: {effective_data_start_date}")
        self.data_start_date = effective_data_start_date
        self.loader.load_data(start_date=effective_data_start_date)
        
        # 2. 初始化模型
        self.model = AlphaGPT().to(ModelConfig.DEVICE)
        self.opt = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(getattr(ModelConfig, "LR", 1e-4)),
            weight_decay=float(getattr(ModelConfig, "WEIGHT_DECAY", 0.01)),
        )
        
        # 3. 杩借釜鏈€浣崇粨鏋?
        self.best_score = -float('inf')
        self.best_formula = None
        self.best_formula_readable = None
        self.best_formula_raw = None
        self.best_formula_raw_readable = None
        self.best_sharpe = 0.0
        self.best_return = 0.0
        
        # 4. 璁板綍鎵€鏈?New King 鍘嗗彶
        self.king_history = []
        
        # 5. 澶氭牱鎬ф睜 (formula_readable -> metrics_dict)
        self.diverse_pool = {}
        
        # 6. Session 绾у洖娴嬬紦瀛?(V3.5: formula_tuple -> result_tuple)
        self.eval_cache = OrderedDict()
        
        print(f"Model vocab size: {self.model.vocab_size}")

    @staticmethod
    def _write_json(path: str, payload: dict) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def _tokens_to_strings(self, tokens: list) -> list:
        """[鍐呴儴鏂规硶] 灏?token ID 鍒楄〃杞崲涓哄瓧绗︿覆鍒楄〃 (浠呯敤浜庤缁冩椂鐨勫嵆鏃惰浆鎹?"""
        vocab = self.model.vocab
        return [vocab[t] if t < len(vocab) else f'?{t}' for t in tokens]
    
    def decode_formula(self, formula: list) -> str:
        """灏嗗叕寮忓瓧绗︿覆鍒楄〃杞崲涓哄彲璇诲瓧绗︿覆"""
        if not formula:
            return ''
        if not isinstance(formula[0], str):
            raise TypeError("decode_formula only accepts string lists, not token IDs")
        return ' '.join(formula)
    
    def _calculate_similarity(self, formula_a: list, formula_b: list) -> float:
        """
        璁＄畻涓や釜鍏紡鐨?Jaccard 鐩镐技搴?(鍩轰簬 Token 闆嗗悎)
        """
        set_a = set(formula_a)
        set_b = set(formula_b)
        intersection = len(set_a.intersection(set_b))
        union = len(set_a.union(set_b))
        return intersection / union if union > 0 else 0.0

    def train(self):
        print("Starting CB Alpha Mining (Multi-Process CPU)...")
        print("=" * 60)
        print("TRAINING CONFIGURATION")
        print(f"   - Steps:       {ModelConfig.TRAIN_STEPS}")
        print(f"   - Batch Size:  {ModelConfig.BATCH_SIZE}")
        print(f"   - Device:      {ModelConfig.DEVICE}")
        print(f"   - Workers:     {os.cpu_count() or 4}")
        print(f"   - Split Date:  {RobustConfig.TRAIN_TEST_SPLIT_DATE}")
        print(f"   - Top-K:       {RobustConfig.TOP_K}")
        print(f"   - Fee Rate:    {RobustConfig.FEE_RATE:.4f} ({RobustConfig.FEE_RATE*100:.2f}% single-side)")
        print(f"   - Min VDays:   {int(RobustConfig.MIN_VALID_DAYS)}")
        print(f"   - Min VRatio:  {float(RobustConfig.MIN_VALID_DAY_RATIO):.1%}")
        print(f"   - Entropy beta:{RobustConfig.ENTROPY_BETA_START} -> {RobustConfig.ENTROPY_BETA_END} (linear decay)")
        print(f"   - Optimizer:   AdamW(lr={ModelConfig.LR}, wd={ModelConfig.WEIGHT_DECAY})")
        print(f"   - Grad Clip:   {ModelConfig.GRAD_CLIP_NORM} (log every {ModelConfig.GRAD_NORM_LOG_INTERVAL} steps)")
        print(f"   - Factors ({len(ModelConfig.INPUT_FEATURES)}): {ModelConfig.INPUT_FEATURES}")
        print(f"   - Operators ({len(OpsRegistry.list_ops())}): {OpsRegistry.list_ops()}")
        print("=" * 60)
        
        start_time = time.time()
        
        # workers 鏁伴噺璁剧疆涓?CPU 鏍稿績鏁?(閫昏緫鏍稿績)
        num_workers = os.cpu_count() or 4
        print(f"Using {num_workers} worker processes")
        
        # 鍑嗗鍏变韩鏁版嵁 (杞负 CPU Tensor)
        cpu_feat = self.loader.feat_tensor.to('cpu')
        cpu_ret = self.loader.target_ret.to('cpu')
        cpu_mask = self.loader.tradable_mask.to('cpu')
        cpu_cs_mask = self.loader.cs_mask.to('cpu')
        split_idx = self.loader.split_idx
        
        # 姝㈢泩浠锋牸鏁版嵁鍑嗗
        # 鏃跺簭瀵归綈璇存槑:
        #   - weights[t] = t鏃ユ敹鐩樻椂鐨勬寔浠撳喅绛?
        #   - target_ret[t] = close[t+1]/close[t] - 1 = 鎸佹湁 t鈫抰+1 鐨勬敹鐩?
        #   - 姝㈢泩妫€鏌ュ彂鐢熷湪鎸佷粨鏈熼棿(t+1鏃ョ洏涓?
        #   - 鍥犳: open_prices[t] 搴斾负 open[t+1], high_prices[t] 搴斾负 high[t+1]
        #   - prev_close[t] = close[t] = 涔板叆浠锋牸
        cpu_open = None
        cpu_high = None
        cpu_prev_close = None
        if RobustConfig.TAKE_PROFIT > 0:
            # 鍙湪鍚敤姝㈢泩鏃跺姞杞戒环鏍兼暟鎹?
            if 'OPEN' in self.loader.raw_data_cache and 'HIGH' in self.loader.raw_data_cache:
                raw_open = self.loader.raw_data_cache['OPEN'].to('cpu')
                raw_high = self.loader.raw_data_cache['HIGH'].to('cpu')
                close = self.loader.raw_data_cache['CLOSE'].to('cpu')
                
                # 鏃跺簭瀵归綈: roll(-1) 浣垮緱 [t] 浣嶇疆瀛樺偍鐨勬槸 t+1 鏃ョ殑浠锋牸
                cpu_open = torch.roll(raw_open, -1, dims=0)
                cpu_high = torch.roll(raw_high, -1, dims=0)
                # 鏈€鍚庝竴琛屾棤鏈夋晥鏁版嵁锛岀疆涓烘瀬澶у€间娇鍏朵笉瑙﹀彂姝㈢泩
                cpu_open[-1] = 1e9
                cpu_high[-1] = 1e9
                
                # prev_close[t] = close[t] = 涔板叆浠凤紙t鏃ユ敹鐩樹环锛?
                cpu_prev_close = close.clone()
                
                print(f"   - Take Profit: {RobustConfig.TAKE_PROFIT:.1%} enabled, price data loaded (time-aligned)")
            else:
                print("   - Warning: TAKE_PROFIT enabled but OPEN/HIGH data not available")
        
        # 鑾峰彇褰撳墠 Config 璺緞 (浼犻€掔粰瀛愯繘绋?
        from .config_loader import get_loaded_config_path
        config_path = get_loaded_config_path()
        
        # 鍚姩杩涚▼姹?
        # 娉ㄦ剰: Windows 涓嬫瘡娆￠兘闇€瑕佸湪杩欓噷浠庡ご鍚姩 executor 姣旇緝瀹夊叏锛屾垨鑰呴暱鏈熸寔鏈?
        # 杩欓噷鎴戜滑閫夋嫨闀挎湡鎸佹湁 executor 涓婁笅鏂?
        with ProcessPoolExecutor(
            max_workers=num_workers, 
            initializer=_init_worker,
            initargs=(
                cpu_feat,
                cpu_ret,
                cpu_mask,
                cpu_cs_mask,
                split_idx,
                cpu_open,
                cpu_high,
                cpu_prev_close,
                config_path,
            )
        ) as executor:
            
            pbar = tqdm(range(ModelConfig.TRAIN_STEPS))
            global_stats = Counter()
            global_struct_reasons = Counter()
            stats_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'training_stats.jsonl')
            
            # [Grammar] 1. 鍑嗗鍏冩暟鎹?(绉昏嚦寰幆澶栦互浼樺寲鎬ц兘)
            is_feat, is_unary, is_binary, net_change = self.model.get_grammar_masks(ModelConfig.DEVICE)
            vocab_size = self.model.vocab_size
            max_len = ModelConfig.MAX_FORMULA_LEN
            max_stack_depth = RobustConfig.MAX_STACK_DEPTH

            token_to_idx = {token: idx for idx, token in enumerate(self.model.vocab)}
            pair_ban_matrix = torch.zeros(
                vocab_size + 1, vocab_size, dtype=torch.bool, device=ModelConfig.DEVICE
            )
            for prev_token, next_token in HARD_FORBIDDEN_SEQUENCES:
                prev_idx = token_to_idx.get(prev_token)
                next_idx = token_to_idx.get(next_token)
                if prev_idx is None or next_idx is None:
                    continue
                pair_ban_matrix[prev_idx + 1, next_idx] = True

            forbidden_terminal_mask = torch.zeros(
                vocab_size, dtype=torch.bool, device=ModelConfig.DEVICE
            )
            for token in FORBIDDEN_TERMINAL_OPS:
                idx = token_to_idx.get(token)
                if idx is not None:
                    forbidden_terminal_mask[idx] = True

            sign_idx = token_to_idx.get("SIGN")
            log_idx = token_to_idx.get("LOG")
            discrete_token_mask = torch.tensor(
                [token in DISCRETE_OPS for token in self.model.vocab],
                dtype=torch.bool,
                device=ModelConfig.DEVICE,
            )
            max_terminal_discrete = int(MAX_TERMINAL_DISCRETE)
            ts_token_mask = torch.tensor(
                [token.startswith("TS_") for token in self.model.vocab],
                dtype=torch.bool,
                device=ModelConfig.DEVICE,
            )
            ts_token_mask_f = ts_token_mask.float()

            density_window = max(1, int(RobustConfig.DENSITY_WINDOW))
            max_ts_in_window = int(RobustConfig.MAX_TS_IN_WINDOW)
            decode_ts_soft_enabled = bool(RobustConfig.DECODE_TS_DENSITY_SOFT_ENABLED)
            decode_reachability_enabled = bool(RobustConfig.DECODE_REACHABILITY_ENABLED)
            decode_lookahead_enabled = bool(RobustConfig.DECODE_LOOKAHEAD_ENABLED)

            ts_penalty_l1 = float(RobustConfig.DECODE_TS_DENSITY_PENALTY_L1)
            ts_penalty_l2 = float(RobustConfig.DECODE_TS_DENSITY_PENALTY_L2)
            ts_penalty_l3 = float(RobustConfig.DECODE_TS_DENSITY_PENALTY_L3)

            terminal_unary_exists = bool((is_unary & ~forbidden_terminal_mask).any().item())
            terminal_binary_exists = bool((is_binary & ~forbidden_terminal_mask).any().item())
            terminal_feat_exists = bool((is_feat & ~forbidden_terminal_mask).any().item())

            safe_token_mask = torch.zeros(vocab_size, dtype=torch.bool, device=ModelConfig.DEVICE)
            for safe_token in ["ABS", "CS_RANK", "TS_MEAN", "TS_STD5", "ADD", "SUB", "MUL", "MAX", "MIN"]:
                safe_idx = token_to_idx.get(safe_token)
                if safe_idx is not None:
                    safe_token_mask[safe_idx] = True
            
            # [V4.1] 鐘舵€佺洃鎺у彉閲?(宸ョ▼鍔犲浐鐗?
            # 1. 鎴愬姛鐜囩獥鍙?(鐢ㄤ簬骞虫粦鎺у埗)
            # [V4.1.1] 鍒濆鍖栦笅璋冧负 0.5锛岄槻姝㈠墠鏈?鍘嗗彶澶ソ"瀵艰嚧涓€绾х喌瑙﹀彂鍙嶅簲杩熼挐
            window_size = 10
            init_bs = ModelConfig.BATCH_SIZE
            metric_pass_seed = max(1, int(init_bs * 0.02))
            hard_pass_rate_history = deque([0.5]*window_size, maxlen=window_size)
            hard_pass_abs_history = deque([int(init_bs * 0.5)]*window_size, maxlen=window_size)
            metric_pass_rate_history = deque([0.02]*window_size, maxlen=window_size)
            metric_pass_abs_history = deque([metric_pass_seed]*window_size, maxlen=window_size)
            sim_pass_rate_history = deque([0.02]*window_size, maxlen=window_size)
            sim_pass_abs_history = deque([metric_pass_seed]*window_size, maxlen=window_size)
            pool_update_rate_history = deque([0.2]*window_size, maxlen=window_size)
            pool_update_abs_history = deque([1.0]*window_size, maxlen=window_size)
            struct_rate_history = deque([0.0]*window_size, maxlen=window_size)
            
            # 2. 鎸佺画鏁呴殰瑙﹀彂鍣?(鐔旀柇閫昏緫)
            struct_failure_strike = 0  # 杩炵画楂?Struct 璁℃暟
            lowvar_failure_strike = 0  # 杩炵画楂?LowVar 璁℃暟
            lowvar_recovery_strike = 0
            saturation_strike = 0
            lowvar_penalty_multiplier = 1.0
            low_reward_std_strike = 0
            steps_since_new_king = 0
            steps_since_pool_update = 0
            
            # 3. 控制器状态
            cool_down_timer = 0
            total_steps = ModelConfig.TRAIN_STEPS
            grad_clip_norm = float(getattr(ModelConfig, "GRAD_CLIP_NORM", 1.0))
            grad_norm_log_interval = max(1, int(getattr(ModelConfig, "GRAD_NORM_LOG_INTERVAL", 20)))
            controller_mode = RobustConfig.ENTROPY_CONTROLLER_MODE
            if controller_mode not in {"hard", "metric", "sim", "pool"}:
                controller_mode = "hard"
            
            for step in pbar:
                step_stats = Counter()
                raw_status_counts = Counter()
                pre_metric_counts = Counter()
                step_valid_day_ratios = []
                # [V4.1] 瀹氫箟涓夌骇鎴愬姛璁℃暟
                counts = {"HardPass": 0, "MetricPass": 0, "SimPass": 0}
                bs = ModelConfig.BATCH_SIZE
                
                # [V4.1] 璁＄畻褰撳墠骞虫粦鎸囨爣
                rolling_hpr = sum(hard_pass_rate_history) / window_size
                rolling_hpa = sum(hard_pass_abs_history) / window_size
                rolling_mpr = sum(metric_pass_rate_history) / window_size
                rolling_mpa = sum(metric_pass_abs_history) / window_size
                rolling_spr = sum(sim_pass_rate_history) / window_size
                rolling_spa = sum(sim_pass_abs_history) / window_size
                rolling_pur = sum(pool_update_rate_history) / window_size
                rolling_pua = sum(pool_update_abs_history) / window_size
                rolling_str = sum(struct_rate_history) / window_size
                
                # [V4.1] 鑷€傚簲鐔垫帶鍒跺洖璺?(Rolling 鐗?
                base_beta = RobustConfig.ENTROPY_BETA_START - (
                    RobustConfig.ENTROPY_BETA_START - RobustConfig.ENTROPY_BETA_END
                ) * (step / total_steps)
                controller_reason = "base"
                
                if cool_down_timer > 0:
                    cool_down_timer -= 1
                    current_beta = getattr(self, '_current_beta_locked', base_beta)
                    controller_reason = "cooldown_lock"
                else:
                    warn_boost = float(RobustConfig.ENTROPY_WARN_BOOST)
                    lock_beta = float(RobustConfig.ENTROPY_LOCK_BETA)
                    lock_steps = int(RobustConfig.ENTROPY_LOCK_STEPS)
                    current_beta = base_beta

                    if controller_mode == "hard":
                        if (
                            rolling_hpr < RobustConfig.CONTROLLER_HARD_FLOOR_RATE
                            or rolling_hpa < RobustConfig.CONTROLLER_HARD_FLOOR_ABS
                        ):
                            current_beta = lock_beta
                            self._current_beta_locked = current_beta
                            cool_down_timer = lock_steps
                            controller_reason = "hard_floor"
                        elif rolling_hpr < RobustConfig.CONTROLLER_HARD_WARN_RATE:
                            current_beta = min(base_beta + warn_boost, 0.08)
                            controller_reason = "hard_warn"
                        else:
                            controller_reason = "hard_ok"
                    elif controller_mode == "metric":
                        if (
                            rolling_mpr < RobustConfig.CONTROLLER_METRIC_FLOOR_RATE
                            or rolling_mpa < RobustConfig.CONTROLLER_METRIC_FLOOR_ABS
                        ):
                            current_beta = lock_beta
                            self._current_beta_locked = current_beta
                            cool_down_timer = lock_steps
                            controller_reason = "metric_floor"
                        elif rolling_mpr < RobustConfig.CONTROLLER_METRIC_WARN_RATE:
                            current_beta = min(base_beta + warn_boost, 0.08)
                            controller_reason = "metric_warn"
                        else:
                            controller_reason = "metric_ok"
                    elif controller_mode == "sim":
                        if (
                            rolling_spr < RobustConfig.CONTROLLER_SIM_FLOOR_RATE
                            or rolling_spa < RobustConfig.CONTROLLER_SIM_FLOOR_ABS
                        ):
                            current_beta = lock_beta
                            self._current_beta_locked = current_beta
                            cool_down_timer = lock_steps
                            controller_reason = "sim_floor"
                        elif rolling_spr < RobustConfig.CONTROLLER_SIM_WARN_RATE:
                            current_beta = min(base_beta + warn_boost, 0.08)
                            controller_reason = "sim_warn"
                        else:
                            controller_reason = "sim_ok"
                    else:
                        if steps_since_pool_update >= RobustConfig.CONTROLLER_POOL_STAGNATION_PATIENCE:
                            current_beta = lock_beta
                            self._current_beta_locked = current_beta
                            cool_down_timer = lock_steps
                            controller_reason = f"pool_stall({steps_since_pool_update})"
                        elif (
                            rolling_pur < RobustConfig.CONTROLLER_POOL_FLOOR_RATE
                            or rolling_pua < RobustConfig.CONTROLLER_POOL_FLOOR_ABS
                        ):
                            current_beta = lock_beta
                            self._current_beta_locked = current_beta
                            cool_down_timer = lock_steps
                            controller_reason = "pool_floor"
                        elif rolling_pur < RobustConfig.CONTROLLER_POOL_WARN_RATE:
                            current_beta = min(base_beta + warn_boost, 0.08)
                            controller_reason = "pool_warn"
                        else:
                            controller_reason = "pool_ok"

                        # pool mode secondary guard: metric pass too low also needs exploration.
                        if (
                            rolling_mpr < RobustConfig.CONTROLLER_METRIC_FLOOR_RATE
                            or rolling_mpa < RobustConfig.CONTROLLER_METRIC_FLOOR_ABS
                        ):
                            current_beta = max(current_beta, min(base_beta + warn_boost, 0.08))
                            controller_reason += "+metric_guard"

                entropy_boost = 0.0
                if steps_since_new_king >= RobustConfig.STAGNATION_PATIENCE:
                    entropy_boost = max(entropy_boost, RobustConfig.STAGNATION_ENTROPY_BOOST)
                if (
                    steps_since_pool_update >= RobustConfig.REWARD_STD_PATIENCE
                    and low_reward_std_strike >= RobustConfig.REWARD_STD_PATIENCE
                ):
                    entropy_boost = max(entropy_boost, RobustConfig.COLLAPSE_ENTROPY_BOOST)
                if entropy_boost > 0:
                    current_beta = min(current_beta + entropy_boost, 0.08)
                    controller_reason += "+boost"

                inp = torch.zeros((bs, 1), dtype=torch.long, device=ModelConfig.DEVICE)

                current_depths = torch.zeros(bs, dtype=torch.long, device=ModelConfig.DEVICE)

                
                log_probs = []
                entropies = []
                tokens_list = []
                prev_actions = torch.full((bs,), -1, dtype=torch.long, device=ModelConfig.DEVICE)
                last_sign_steps = torch.full((bs,), -10_000, dtype=torch.long, device=ModelConfig.DEVICE)
                ts_recent_flags = deque()
                ts_recent_count = torch.zeros(bs, dtype=torch.long, device=ModelConfig.DEVICE)
                terminal_discrete_tail = torch.zeros(bs, dtype=torch.long, device=ModelConfig.DEVICE)
                empty_sample_hit = torch.zeros(bs, dtype=torch.bool, device=ModelConfig.DEVICE)
                decode_mask_empty_count = 0
                decode_fallback_force_finish_count = 0
                decode_fallback_repair_count = 0
                decode_fallback_safe_count = 0
                decode_fallback_last_resort_count = 0
                fallback_reason_counter = Counter()

                
                # 鑷洖褰掔敓鎴愬叕寮?(鍦?Main Process GPU 涓婅繘琛?
                for gen_step in range(ModelConfig.MAX_FORMULA_LEN):
                    logits, _ = self.model(inp)
                    R = max_len - 1 - gen_step

                    # [Grammar] 2. Action Masking
                    mask = torch.ones(bs, vocab_size, dtype=torch.bool, device=ModelConfig.DEVICE)
                    
                    # Rule 1: D < 1 -> Ban All Ops (Underflow)
                    mask &= ~((current_depths < 1).unsqueeze(1) & (is_unary | is_binary))
                    
                    # Rule 2: D < 2 -> Ban Binary (Underflow)
                    mask &= ~((current_depths < 2).unsqueeze(1) & is_binary)
                    
                    # Rule 3: R <= D - 1 -> Ban Feature (Overrun, 鏃犳硶褰掔害)
                    # 鍓╀綑姝ユ暟 R = (Total - 1) - current_step
                    # e.g. Total=12. Step=0, R=11. Step=11, R=0.
                    # R already computed above
                    mask &= ~((R <= current_depths - 1).unsqueeze(1) & is_feat)
                    
                    # Rule 4: R < D - 1 -> Ban Unary (Force Binary Reduce)
                    mask &= ~((R < current_depths - 1).unsqueeze(1) & is_unary)
                    
                    # Rule 5: D >= MAX -> Ban Feature (Stack Limit)
                    mask &= ~((current_depths >= max_stack_depth).unsqueeze(1) & is_feat)
                    
                    # Rule 6: hard forbidden token pairs
                    if gen_step > 0:
                        pair_ban = pair_ban_matrix[(prev_actions + 1).long()]
                        mask &= ~pair_ban

                    # Rule 7: SIGN -> LOG proximity hard ban
                    if sign_idx is not None and log_idx is not None and MAX_SIGN_LOG_DISTANCE >= 1:
                        sign_recent = (last_sign_steps >= 0) & (
                            (gen_step - last_sign_steps) <= MAX_SIGN_LOG_DISTANCE
                        )
                        if sign_recent.any():
                            mask[sign_recent, log_idx] = False

                    # Rule 8: terminal operator ban at final position
                    if gen_step == max_len - 1:
                        mask &= ~forbidden_terminal_mask.unsqueeze(0)
                        if max_terminal_discrete <= 0:
                            mask &= ~discrete_token_mask.unsqueeze(0)
                        else:
                            too_many_terminal_discrete = terminal_discrete_tail >= max_terminal_discrete
                            if too_many_terminal_discrete.any():
                                mask[too_many_terminal_discrete] &= ~discrete_token_mask

                    # Rule 9: reachability bounds check
                    if decode_reachability_enabled:
                        depth_after = current_depths.unsqueeze(1) + net_change.unsqueeze(0)
                        remain_after = max_len - (gen_step + 1)
                        min_reachable = depth_after - remain_after
                        max_reachable = depth_after + remain_after
                        mask &= (depth_after >= 1) & (min_reachable <= 1) & (max_reachable >= 1)

                    # Rule 10: one-step lookahead (avoid dead-end at the penultimate step)
                    if decode_lookahead_enabled and (max_len - (gen_step + 1) == 1):
                        depth_after = current_depths.unsqueeze(1) + net_change.unsqueeze(0)
                        can_finish_next = torch.zeros_like(depth_after, dtype=torch.bool)
                        if terminal_unary_exists:
                            can_finish_next |= (depth_after == 1)
                        if terminal_binary_exists:
                            can_finish_next |= (depth_after == 2)
                        if terminal_feat_exists:
                            can_finish_next |= (depth_after == 0)
                        mask &= can_finish_next

                    # Deterministic fallback for all-masked rows
                    empty_rows = ~mask.any(dim=1)
                    if empty_rows.any():
                        empty_indices = empty_rows.nonzero(as_tuple=False).squeeze(1)
                        decode_mask_empty_count += int(empty_indices.numel())
                        empty_sample_hit[empty_rows] = True
                        for row_idx in empty_indices.tolist():
                            row_mask = mask[row_idx]
                            row_mask.fill_(False)
                            depth_val = int(current_depths[row_idx].item())
                            remain_after = max_len - (gen_step + 1)
                            tail_discrete_val = int(terminal_discrete_tail[row_idx].item())
                            ban_terminal_discrete = (
                                remain_after == 0 and (
                                    max_terminal_discrete <= 0 or tail_discrete_val >= max_terminal_discrete
                                )
                            )
                            used_fallback = False

                            # Priority 1: if this is terminal step, force a finish-capable action.
                            if remain_after == 0:
                                if depth_val == 1:
                                    candidate = is_unary & ~forbidden_terminal_mask
                                elif depth_val == 2:
                                    candidate = is_binary & ~forbidden_terminal_mask
                                else:
                                    candidate = torch.zeros_like(row_mask)
                                if ban_terminal_discrete:
                                    candidate &= ~discrete_token_mask
                                if candidate.any():
                                    row_mask |= candidate
                                    decode_fallback_force_finish_count += 1
                                    fallback_reason_counter["force_finish"] += 1
                                    used_fallback = True

                            # Priority 2: repair stack state.
                            if not used_fallback:
                                if depth_val >= 2:
                                    candidate = is_binary.clone()
                                    if gen_step == max_len - 1:
                                        candidate &= ~forbidden_terminal_mask
                                    if candidate.any():
                                        row_mask |= candidate
                                        decode_fallback_repair_count += 1
                                        fallback_reason_counter["repair_binary"] += 1
                                        used_fallback = True
                                if (not used_fallback) and depth_val <= 0:
                                    candidate = is_feat.clone()
                                    if candidate.any():
                                        row_mask |= candidate
                                        decode_fallback_repair_count += 1
                                        fallback_reason_counter["repair_feature"] += 1
                                        used_fallback = True
                                if (not used_fallback) and depth_val == 1:
                                    candidate = is_unary.clone()
                                    if gen_step == max_len - 1:
                                        candidate &= ~forbidden_terminal_mask
                                    if ban_terminal_discrete:
                                        candidate &= ~discrete_token_mask
                                    if candidate.any():
                                        row_mask |= candidate
                                        decode_fallback_repair_count += 1
                                        fallback_reason_counter["repair_unary"] += 1
                                        used_fallback = True

                            # Priority 3: safe token list.
                            if not used_fallback:
                                candidate = safe_token_mask.clone()
                                if depth_val < 1:
                                    candidate &= is_feat
                                elif depth_val < 2:
                                    candidate &= ~is_binary
                                if gen_step == max_len - 1:
                                    candidate &= ~forbidden_terminal_mask
                                if ban_terminal_discrete:
                                    candidate &= ~discrete_token_mask
                                if candidate.any():
                                    row_mask |= candidate
                                    decode_fallback_safe_count += 1
                                    fallback_reason_counter["safe_list"] += 1
                                    used_fallback = True

                            # Last resort.
                            if not used_fallback:
                                row_mask[:] = True
                                if gen_step == max_len - 1:
                                    row_mask &= ~forbidden_terminal_mask
                                if ban_terminal_discrete:
                                    row_mask &= ~discrete_token_mask
                                if not row_mask.any():
                                    row_mask[:] = True
                                decode_fallback_last_resort_count += 1
                                fallback_reason_counter["last_resort"] += 1

                    # TS density soft penalty: graded logit down-weight (not hard mask).
                    if decode_ts_soft_enabled and max_ts_in_window >= 0:
                        projected_ts = ts_recent_count + 1
                        excess = projected_ts - max_ts_in_window
                        ts_penalty = torch.zeros(bs, dtype=logits.dtype, device=ModelConfig.DEVICE)
                        ts_penalty = torch.where(excess == 1, ts_penalty + ts_penalty_l1, ts_penalty)
                        ts_penalty = torch.where(excess == 2, ts_penalty + ts_penalty_l2, ts_penalty)
                        ts_penalty = torch.where(excess >= 3, ts_penalty + ts_penalty_l3, ts_penalty)
                        logits = logits - ts_penalty.unsqueeze(1) * ts_token_mask_f.unsqueeze(0)

                    # Apply Mask
                    logits = logits.masked_fill(~mask, -1e9)

                    dist = Categorical(logits=logits)

                    action = dist.sample()
                    
                    # [Grammar] 3. 鏇存柊娣卞害
                    current_depths += net_change[action]
                    prev_actions = action

                    if sign_idx is not None:
                        sign_pos = torch.full_like(last_sign_steps, gen_step)
                        last_sign_steps = torch.where(action == sign_idx, sign_pos, last_sign_steps)

                    is_discrete_action = discrete_token_mask[action].long()
                    terminal_discrete_tail = torch.where(
                        is_discrete_action > 0,
                        terminal_discrete_tail + 1,
                        torch.zeros_like(terminal_discrete_tail),
                    )

                    if density_window > 1:
                        current_is_ts = ts_token_mask[action].long()
                        ts_recent_flags.append(current_is_ts)
                        ts_recent_count = ts_recent_count + current_is_ts
                        if len(ts_recent_flags) > density_window - 1:
                            ts_recent_count = ts_recent_count - ts_recent_flags.popleft()

                    log_probs.append(dist.log_prob(action))
                    entropies.append(dist.entropy())
                    tokens_list.append(action)
                    inp = torch.cat([inp, action.unsqueeze(1)], dim=1)
                
                seqs = torch.stack(tokens_list, dim=1)
                
                # 鍑嗗浠诲姟: 绔嬪嵆灏?token ID 杞负瀛楃涓诧紝娑堥櫎瑙ｇ爜涓嶄竴鑷撮闄?
                raw_formula_list = [self._tokens_to_strings(seq.tolist()) for seq in seqs]
                formula_list = [simplify_formula(formula) for formula in raw_formula_list]
                formula_keys = [formula_to_canonical_key(formula) for formula in formula_list]
                
                # [V3.5] 鎵规鍘婚噸涓?LRU 缂撳瓨 (鏍稿績鎻愰€熼€昏緫)
                unique_formula_to_indices = {}
                key_to_formula = {}
                for idx, (f, f_key) in enumerate(zip(formula_list, formula_keys)):
                    if f_key not in unique_formula_to_indices:
                        unique_formula_to_indices[f_key] = []
                        key_to_formula[f_key] = f
                    unique_formula_to_indices[f_key].append(idx)
                
                num_unique_gen = len(unique_formula_to_indices)
                
                # 璇嗗埆涓嶅湪缂撳瓨涓殑鍞竴鍏紡
                to_eval_formulas = []
                f_idx_to_eval = {}
                for f_key in unique_formula_to_indices:
                    if f_key not in self.eval_cache:
                        f_idx_to_eval[f_key] = len(to_eval_formulas)
                        to_eval_formulas.append(key_to_formula[f_key])
                
                num_to_eval = len(to_eval_formulas)
                
                # 浠呭洖娴嬩粠鏈杩囦笖褰撳墠鎵规鍞竴鐨勫叕寮?
                if to_eval_formulas:
                    new_results = list(executor.map(_worker_eval, to_eval_formulas))
                    for f_key, eval_idx in f_idx_to_eval.items():
                        # LRU Update
                        if f_key in self.eval_cache:
                            self.eval_cache.move_to_end(f_key)
                        self.eval_cache[f_key] = new_results[eval_idx]
                    
                    # [V3.5] Cache 瀹归噺绠＄悊 (LRU 娣樻卑)
                    max_cache = RobustConfig.CACHE_MAX_SIZE
                    if len(self.eval_cache) > max_cache:
                        # 娣樻卑鏈€鏃╃殑 20%
                        for _ in range(int(max_cache * 0.2)):
                            self.eval_cache.popitem(last=False)
                
                # 缁勮 512 涓粨鏋?
                results = []
                for f_key in formula_keys:
                    res_tuple = self.eval_cache[f_key]
                    results.append(res_tuple)
                    # 姣忔鍛戒腑閮界Щ鍒版湯灏?(LRU)
                    self.eval_cache.move_to_end(f_key)
                
                # [V3.5] 鎻愰€熸寚鏍囪绠?
                batch_hit_rate = (bs - num_unique_gen) / bs
                cache_hit_rate = (num_unique_gen - num_to_eval) / bs
                uniq_rate_gen = num_unique_gen / bs
                
                rewards_list = []
                step_gaps = []
                step_struct_reasons = Counter()
                step_new_king = 0
                step_pool_updates = 0
                step_pass_candidates = []

                
                # 鑱氬悎缁撴灉
                for i, (rew, best_info, status, detail) in enumerate(results):
                    raw_formula = raw_formula_list[i]
                    simplified_formula = formula_list[i]
                    final_status = status
                    raw_status_counts[status] += 1
                    if status == "STRUCT_INVALID":
                        pre_metric_counts["validator_fail"] += 1
                    elif status in ("EXEC_ERR", "EXEC_NONE"):
                        pre_metric_counts["exec_fail"] += 1
                    elif status == "LOW_VARIANCE":
                        pre_metric_counts["lowvar_fail"] += 1
                    elif status.startswith("METRIC_"):
                        pre_metric_counts["metric_fail"] += 1
                    elif status == "PASS":
                        pre_metric_counts["post_metric_pass"] += 1
                    else:
                        pre_metric_counts["other"] += 1
                    
                    # [V4.1] 涓夌骇鎴愬姛瀹氫箟
                    is_hard_pass = status not in ["STRUCT_INVALID", "EXEC_ERR", "EXEC_NONE", "LOW_VARIANCE"]
                    if is_hard_pass:
                        counts["HardPass"] += 1
                        if status == "PASS":
                            counts["MetricPass"] += 1
                    
                    # [V4.1] 鍔ㄦ€佹儵缃氬€嶇巼搴旂敤 (閽堝椤藉浐 LowVar)
                    if status == "LOW_VARIANCE":
                        rew = rew * lowvar_penalty_multiplier
                    
                    rewards_list.append(rew)
                    step_stats[status] += 1
                    
                    if status == "STRUCT_INVALID":
                        global_struct_reasons[detail] += 1
                        step_struct_reasons[detail] += 1
                    
                    # 鎻愬彇 Gap 鎸囨爣
                    if isinstance(detail, str) and "gap=" in detail:
                        match = re.search(r"gap=([-+]?\d*\.?\d+)", detail)
                        if match:
                            try:
                                step_gaps.append(float(match.group(1)))
                            except Exception:
                                pass
                    
                    score_val = None
                    pool_action = "NA"
                    is_new_king = False
                    if best_info:
                        # V2.2: best_info 鐜板寘鍚?(score, annualized_ret, sharpe_all, formula, metrics)
                        score_val, ret_val, sharpe_val, formula_str, metrics = best_info
                        step_valid_day_ratios.append(float(metrics.get('valid_day_ratio', 0.0)))
                        simplified_readable = self.decode_formula(formula_str)
                        raw_readable = self.decode_formula(raw_formula)
                        
                        # 浼樺寲: 鍙湁鎻愬崌瓒呰繃闃堝€兼墠瑙嗕负 New King锛屽噺灏?I/O 闃诲
                        if score_val > self.best_score + RobustConfig.MIN_SCORE_IMPROVEMENT:
                            self.best_score = score_val
                            self.best_formula = formula_str  # 鐜板湪鏄瓧绗︿覆鍒楄〃
                            self.best_formula_readable = simplified_readable
                            self.best_formula_raw = raw_formula
                            self.best_formula_raw_readable = raw_readable
                            self.best_sharpe = sharpe_val
                            self.best_return = ret_val
                            step_new_king += 1
                            is_new_king = True
                            
                            # 璁板綍鍒板巻鍙?(V2.2: 鍖呭惈绋冲仴鎬ф寚鏍?+ 骞村寲鏀剁泭 + IC/IR)
                            king_num = len(self.king_history) + 1
                            self.king_history.append({
                                'step': step,
                                'score': score_val,
                                'sharpe': sharpe_val,
                                'sharpe_train': metrics.get('sharpe_train', 0),
                                'sharpe_val': metrics.get('sharpe_val', 0),
                                'sharpe_train_valid_days': metrics.get('sharpe_train_valid_days', 0),
                                'sharpe_val_valid_days': metrics.get('sharpe_val_valid_days', 0),
                                'sharpe_all_valid_days': metrics.get('sharpe_all_valid_days', 0),
                                'max_drawdown': metrics.get('max_drawdown', 0),
                                'stability': metrics.get('stability_metric', 0),
                                'annualized_ret': ret_val,  # 骞村寲鏀剁泭鐜?
                                'annualized_ret_valid_days': metrics.get('annualized_ret_valid_days', 0),
                                'valid_signal_days': metrics.get('valid_signal_days', 0),
                                'valid_day_ratio': metrics.get('valid_day_ratio', 0),
                                # IC/IR 鎸囨爣
                                'ic_mean': metrics.get('ic_mean', 0),
                                'ic_std': metrics.get('ic_std', 0),
                                'ic_ir': metrics.get('ic_ir'),
                                'ic_ir_annual': metrics.get('ic_ir_annual'),
                                'valid_ic_days': metrics.get('valid_ic_days', 0),
                                'skipped_ic_days': metrics.get('skipped_ic_days', 0),
                                'formula': formula_str,
                                'readable': self.best_formula_readable,
                                'raw_formula': raw_formula,
                                'raw_readable': raw_readable,
                            })
                            
                            # 淇濆瓨浜ゆ槗缁嗚妭鍒扮嫭绔嬫枃浠?
                            self._save_king_trades(king_num, formula_str, score_val, sharpe_val, ret_val)
                            
                            # IC/IR 瀹夊叏鏍煎紡鍖?
                            ic_val = metrics.get('ic_mean', 0)
                            ir_val = metrics.get('ic_ir')
                            ir_str = f"{ir_val:.2f}" if ir_val is not None else "None"
                            
                            tqdm.write(
                                f"[!] New King #{king_num}: Score {score_val:.2f} | "
                                f"SharpeFull T/V {metrics.get('sharpe_train', 0):.2f}/{metrics.get('sharpe_val', 0):.2f} | "
                                f"SharpeSparse T/V {metrics.get('sharpe_train_valid_days', 0):.2f}/{metrics.get('sharpe_val_valid_days', 0):.2f} | "
                                f"ValidRatio {metrics.get('valid_day_ratio', 0):.1%} | "
                                f"IC {ic_val:.3f} (IR {ir_str}) | MDD {metrics.get('max_drawdown', 0):.1%} | "
                                f"{self.best_formula_readable}"
                            )

                        # V2.3: 鏀堕泦澶氭牱鎬у叕寮?(Diversity Pool)
                        # 浠呭綋鍒嗘暟瓒冲楂樹笖鍏紡鐙壒鏃跺叆姹?
                        if status == "PASS" and score_val > 0:
                            pool_action = "PASS_POS"
                            readable = simplified_readable
                            new_result = {
                                'step': step,  # 璁板綍浜х敓姝ユ暟
                                'score': score_val,
                                'sharpe': sharpe_val,
                                'annualized_ret': ret_val,
                                'valid_day_ratio': metrics.get('valid_day_ratio', 0),
                                'formula': formula_str,
                                'readable': readable,
                                'raw_formula': raw_formula,
                                'raw_readable': raw_readable,
                            }
                            
                            # V2.3: Jaccard 鐩镐技搴﹁繃婊?
                            # 妫€鏌ヤ笌姹犱腑鐜版湁鍏紡鐨勭浉浼煎害
                            similar_key = None
                            for pool_key, pool_data in self.diverse_pool.items():
                                similarity = self._calculate_similarity(formula_str, pool_data['formula'])
                                if similarity > RobustConfig.JACCARD_THRESHOLD:  # 鐩镐技搴﹂槇鍊?
                                    similar_key = pool_key
                                    break
                            
                            # 鍏ユ睜閫昏緫
                            if similar_key is None:
                                # Case 1: 涓庢睜涓棤鐩镐技鍏紡 -> 姝ｅ父鍏ユ睜
                                if readable not in self.diverse_pool:
                                    if len(self.diverse_pool) < RobustConfig.DIVERSITY_POOL_SIZE:
                                        self.diverse_pool[readable] = new_result
                                        step_pool_updates += 1
                                        pool_action = "POOL_ADD"
                                    else:
                                        # 姹犳弧: 鏇挎崲鏈€浣庡垎鍏紡 (濡傛灉鏂板叕寮忔洿濂?
                                        min_key = min(self.diverse_pool, key=lambda k: self.diverse_pool[k]['score'])
                                        if score_val > self.diverse_pool[min_key]['score']:
                                            del self.diverse_pool[min_key]
                                            self.diverse_pool[readable] = new_result
                                            step_pool_updates += 1
                                            pool_action = "POOL_REPLACE_MIN"
                                        else:
                                            pool_action = "POOL_SKIP_LOW"
                                else:
                                    pool_action = "POOL_DUP_KEY"
                            else:
                                # Case 2: 涓庢睜涓煇鍏紡楂樺害鐩镐技 -> 浠呭綋鍒嗘暟鏄捐憲鏇撮珮 (+10%) 鏃舵浛鎹?
                                similar_score = self.diverse_pool[similar_key]['score']
                                if score_val > similar_score * 1.1:  # 闇€瑕侀珮鍑?10%
                                    del self.diverse_pool[similar_key]
                                    self.diverse_pool[readable] = new_result
                                    step_pool_updates += 1
                                    final_status = "SIM_REPLACE"
                                    pool_action = "POOL_SIM_REPLACE"
                                else:
                                    final_status = "SIM_REJECT"
                                    # [V3.5] 应用相似度拒绝惩罚，避免躲进冗余区
                                    rewards_list[i] = RobustConfig.PENALTY_SIM
                                    pool_action = "POOL_SIM_REJECT"
                        elif status == "PASS":
                            pool_action = "PASS_NONPOS"
                    
                    if status == "PASS" and score_val is not None:
                        step_pass_candidates.append({
                            "score": float(score_val),
                            "pool_action": pool_action,
                            "final_status": final_status,
                            "is_new_king": bool(is_new_king),
                            "formula": simplified_formula,
                            "readable": self.decode_formula(simplified_formula),
                            "raw_formula": raw_formula,
                            "raw_readable": self.decode_formula(raw_formula),
                        })

                    # [V4.1.2] SimPass 瀹氫箟淇: 閫氱敤鐨?MetricPass 涓旀湭琚?SIM_REJECT (鍖呭惈 SIM_REPLACE)
                    if final_status != "SIM_REJECT" and is_hard_pass and status == "PASS":
                        counts["SimPass"] += 1

                    
                    step_stats[final_status] += 1
                    global_stats[final_status] += 1

                if step_new_king > 0:
                    steps_since_new_king = 0
                else:
                    steps_since_new_king += 1
                if step_pool_updates > 0:
                    steps_since_pool_update = 0
                else:
                    steps_since_pool_update += 1

                # [V4.1.1] 濂栧姳楗卞拰鐩戞帶 (Standard Deviation)
                rewards_tensor = torch.tensor(rewards_list, dtype=torch.float)
                reward_std = rewards_tensor.std().item()
                if reward_std < RobustConfig.REWARD_STD_FLOOR:
                    low_reward_std_strike += 1
                else:
                    low_reward_std_strike = 0
                # 淇: 鐩戞帶 MetricFail 鐨?Std 鑰屼笉鏄?MetricPass (PASS)
                metric_fail_rewards = [
                    float(r) for i, r in enumerate(rewards_list)
                    if str(results[i][2]).startswith("METRIC_")
                ]
                if metric_fail_rewards:
                    metric_fail_reward_avg = float(np.mean(metric_fail_rewards))
                    metric_fail_reward_p90 = float(np.percentile(metric_fail_rewards, 90))
                    metric_fail_std = float(np.std(metric_fail_rewards))
                else:
                    metric_fail_reward_avg = 0.0
                    metric_fail_reward_p90 = 0.0
                    metric_fail_std = 0.0
                metric_gap_avg = float(np.mean(step_gaps)) if step_gaps else 0.0
                metric_gap_p90 = float(np.percentile(step_gaps, 90)) if step_gaps else 0.0
                avg_valid_day_ratio = float(np.mean(step_valid_day_ratios)) if step_valid_day_ratios else 0.0
                valid_formula_rate = 1.0 - (raw_status_counts["STRUCT_INVALID"] / bs)
                decode_mask_empty_rate = decode_mask_empty_count / float(bs * max_len)
                decode_mask_empty_sample_count = int(empty_sample_hit.sum().item())
                decode_mask_empty_sample_rate = decode_mask_empty_sample_count / float(bs)
                fallback_reason_topk = fallback_reason_counter.most_common(5)
                struct_reason_top3 = step_struct_reasons.most_common(3)
                reason_breakdown_before_metric = {
                    "validator_fail": int(pre_metric_counts["validator_fail"]),
                    "exec_fail": int(pre_metric_counts["exec_fail"]),
                    "lowvar_fail": int(pre_metric_counts["lowvar_fail"]),
                    "metric_fail": int(pre_metric_counts["metric_fail"]),
                    "post_metric_pass": int(pre_metric_counts["post_metric_pass"]),
                    "other": int(pre_metric_counts["other"]),
                }
                near_king_candidates = [
                    c for c in step_pass_candidates if not c["is_new_king"]
                ]
                near_king_candidates.sort(key=lambda x: x["score"], reverse=True)
                near_king_top3 = []
                for c in near_king_candidates[:3]:
                    near_king_top3.append({
                        "score": c["score"],
                        "best_gap": float(self.best_score - c["score"]),
                        "pool_action": c["pool_action"],
                        "final_status": c["final_status"],
                        "readable": c["readable"],
                        "raw_readable": c["raw_readable"],
                    })

                log_entry = {
                    "step": step,
                    "stats": dict(step_stats),
                    "raw_status_counts": dict(raw_status_counts),
                    "controller_mode": controller_mode,
                    "controller_reason": controller_reason,
                    "rolling_hpr": rolling_hpr,
                    "rolling_mpr": rolling_mpr,
                    "rolling_spr": rolling_spr,
                    "rolling_pur": rolling_pur,
                    "reward_std": reward_std,
                    "metric_fail_std": metric_fail_std, # 鏂板 MetricFail Std
                    "metric_fail_reward_avg": metric_fail_reward_avg,
                    "metric_fail_reward_p90": metric_fail_reward_p90,
                    "metric_gap_avg": metric_gap_avg,
                    "metric_gap_p90": metric_gap_p90,
                    "avg_valid_day_ratio": avg_valid_day_ratio,
                    "valid_formula_rate": valid_formula_rate,
                    "decode_mask_empty_count": decode_mask_empty_count,
                    "decode_mask_empty_rate": decode_mask_empty_rate,
                    "decode_mask_empty_sample_count": decode_mask_empty_sample_count,
                    "decode_mask_empty_sample_rate": decode_mask_empty_sample_rate,
                    "decode_fallback_force_finish_count": decode_fallback_force_finish_count,
                    "decode_fallback_repair_count": decode_fallback_repair_count,
                    "decode_fallback_safe_count": decode_fallback_safe_count,
                    "decode_fallback_last_resort_count": decode_fallback_last_resort_count,
                    "fallback_reason_topk": fallback_reason_topk,
                    "struct_reason_top3": struct_reason_top3,
                    "reason_breakdown_before_metric": reason_breakdown_before_metric,
                    "near_king_top3": near_king_top3,
                    "entropy_boost": entropy_boost,
                    "steps_since_new_king": steps_since_new_king,
                    "steps_since_pool_update": steps_since_pool_update,
                    "low_reward_std_strike": low_reward_std_strike,
                }

                if step % 20 == 0:
                    total = bs
                    if step_gaps:
                        avg_gap = np.mean(step_gaps)
                        p50_gap = np.percentile(step_gaps, 50)
                        p90_gap = np.percentile(step_gaps, 90)
                    else:
                        avg_gap = p50_gap = p90_gap = 0.0
                    
                    # [V4.1] 鍔ㄥ姏瀛﹁娴?2.0++: 鍖呭惈 FailAbs 涓庡垎绾?Pass
                    struct_abs = raw_status_counts['STRUCT_INVALID']
                    lowvar_abs = raw_status_counts['LOW_VARIANCE']
                    metric_abs = sum(raw_status_counts[s] for s in raw_status_counts if 'METRIC' in s)
                    sim_abs = step_stats['SIM_REJECT']
                    
                    # [V4.1.1] 璁＄畻 SimFailShare 涓?TopFail
                    # [V4.1.2] 淇: 鍒嗘瘝涓?MetricPass (鍦ㄨ繖浜?Good Ones 閲屾湁澶氬皯鏄噸澶嶇殑)
                    sim_fs_denom = counts['MetricPass']
                    sim_fail_share = sim_abs / sim_fs_denom if sim_fs_denom > 0 else 0.0
                    top_fails = struct_reason_top3
                    top_fail_str = " | ".join([f"{k}:{v}" for k, v in top_fails])
                    
                    msg = (
                        f"[Step {step}] "
                        f"H/M/S_Pass: {counts['HardPass']}/{counts['MetricPass']}/{counts['SimPass']} | "
                        f"Roll[H/M/S/P] {rolling_hpr:.1%}/{rolling_mpr:.1%}/{rolling_spr:.1%}/{rolling_pur:.1%} | "
                        f"Ctl {controller_mode}:{controller_reason} | "
                        f"FailAbs[S:{struct_abs}, L:{lowvar_abs}, M:{metric_abs}, R:{sim_abs}] | "
                        f"Gap(avg/p50/p90) {avg_gap:.2f}/{p50_gap:.2f}/{p90_gap:.2f} | "
                        f"VRatio(avg) {avg_valid_day_ratio:.1%} | "
                        f"RStd {reward_std:.2f} | "
                        f"MFailRew(avg/p90/std) {metric_fail_reward_avg:.2f}/{metric_fail_reward_p90:.2f}/{metric_fail_std:.2f} | "
                        f"Valid {valid_formula_rate:.1%} | "
                        f"MaskEmpty[T:{decode_mask_empty_rate:.1%},S:{decode_mask_empty_sample_rate:.1%}] | "
                        f"SimFS {sim_fail_share:.1%} | B:{current_beta:.4f} "
                        f"(+{entropy_boost:.3f}) | KWait:{steps_since_new_king}"
                    )
                    tqdm.write(msg)
                    if near_king_top3:
                        near_str = " | ".join(
                            [
                                f"{c['score']:.2f}(gap {c['best_gap']:.2f}, {c['pool_action']})"
                                for c in near_king_top3
                            ]
                        )
                        tqdm.write(f"   NearKingTop3: {near_str}")
                    else:
                        tqdm.write("   NearKingTop3: none")
                    if top_fail_str:
                        tqdm.write(f"   TopFail: {top_fail_str}")
                    if fallback_reason_topk:
                        fallback_str = " | ".join([f"{k}:{v}" for k, v in fallback_reason_topk])
                        tqdm.write(f"   FallbackTop: {fallback_str}")
                
                # [V4.1] 鏇存柊婊氬姩绐楀彛涓庢寔缁晠闅滆Е鍙戝櫒
                hpr = counts['HardPass'] / bs
                mpr = counts['MetricPass'] / bs
                spr = counts['SimPass'] / bs
                pool_update_rate = 1.0 if step_pool_updates > 0 else 0.0
                hard_pass_rate_history.append(hpr)
                hard_pass_abs_history.append(counts['HardPass'])
                metric_pass_rate_history.append(mpr)
                metric_pass_abs_history.append(counts['MetricPass'])
                sim_pass_rate_history.append(spr)
                sim_pass_abs_history.append(counts['SimPass'])
                pool_update_rate_history.append(pool_update_rate)
                pool_update_abs_history.append(float(step_pool_updates))
                struct_rate = raw_status_counts['STRUCT_INVALID'] / bs
                struct_rate_history.append(struct_rate)
                
                # 鐔旀柇瑙勫垯 1: 缁撴瀯鍧嶇缉 (30姝?> 90%)
                if struct_rate > 0.9:
                    struct_failure_strike += 1
                else:
                    struct_failure_strike = 0
                
                if struct_failure_strike >= 30:
                    tqdm.write(">>> [CRITICAL] Struct Collapse detected! Entering Reinforcement Mode.")
                    current_beta = 0.06
                    self._current_beta_locked = current_beta
                    cool_down_timer = 20
                    struct_failure_strike = 0
                
                # 鐔旀柇瑙勫垯 2: 浣庢柟宸潔缂?(50姝?> 70%)
                if (step_stats['LOW_VARIANCE'] / bs) > 0.7:
                    lowvar_failure_strike += 1
                else:
                    lowvar_failure_strike = max(0, lowvar_failure_strike - 1)
                
                if lowvar_failure_strike >= 50:
                    tqdm.write(">>> [WARNING] Persistent LowVar detected. Hardening penalty.")
                    lowvar_penalty_multiplier = 1.15
                    lowvar_failure_strike = 0
                
                # [V4.1.1] 鐔旀柇瑙勫垯 3: 浣庢柟宸仮澶?
                if (step_stats['LOW_VARIANCE'] / bs) < 0.1:
                    lowvar_recovery_strike += 1
                else:
                    lowvar_recovery_strike = 0
                if lowvar_recovery_strike >= 20 and lowvar_penalty_multiplier > 1.0:
                    tqdm.write(">>> [INFO] LowVar recovered. Normalizing penalty.")
                    lowvar_penalty_multiplier = 1.0
                    lowvar_recovery_strike = 0

                # [V4.1.1] 鐔旀柇瑙勫垯 4: 濂栧姳楗卞拰鍛婅
                if reward_std < RobustConfig.REWARD_STD_FLOOR:
                    saturation_strike += 1
                else:
                    saturation_strike = 0
                if saturation_strike >= RobustConfig.REWARD_STD_PATIENCE:
                    tqdm.write(
                        f">>> [LOG] Reward Saturation Detected "
                        f"(RStd {reward_std:.3f} < {RobustConfig.REWARD_STD_FLOOR:.3f})."
                    )
                    saturation_strike = 0

                rewards = torch.tensor(rewards_list, device=ModelConfig.DEVICE)

                
                # 浼樺娍鍑芥暟褰掍竴鍖?
                adv = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
                if reward_std < RobustConfig.REWARD_STD_FLOOR and RobustConfig.ADV_NOISE_STD > 0:
                    adv = adv + torch.randn_like(adv) * RobustConfig.ADV_NOISE_STD
                
                # Loss & Update
                loss = 0
                for t in range(len(log_probs)):
                    loss += -log_probs[t] * adv
                
                loss = loss.mean()
                
                # V3.5: 浣跨敤甯︾簿鐐兼帶鍒跺洖璺殑鐔垫鍒欏寲
                avg_entropy = torch.stack(entropies).mean()
                loss = loss - current_beta * avg_entropy
                
                self.opt.zero_grad()
                loss.backward()
                if grad_clip_norm > 0:
                    grad_norm = float(
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=grad_clip_norm)
                    )
                else:
                    grad_sq = 0.0
                    for p in self.model.parameters():
                        if p.grad is not None:
                            grad_sq += float(p.grad.detach().norm(2).item() ** 2)
                    grad_norm = grad_sq ** 0.5
                self.opt.step()

                log_entry["grad_norm"] = grad_norm
                log_entry["grad_clip_norm"] = grad_clip_norm
                log_entry["timestamp"] = time.time()
                with open(stats_path, 'a', encoding='utf-8') as stats_file:
                    stats_file.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

                if step % grad_norm_log_interval == 0:
                    tqdm.write(f"   GradNorm {grad_norm:.3f} | Clip {grad_clip_norm:.3f}")
                
                pbar.set_postfix({
                    'AvgRew': f'{rewards.mean().item():.2f}',
                    'Best': f'{self.best_score:.2f}',
                    'Ent': f'{avg_entropy.item():.2f}',
                    '尾': f'{current_beta:.3f}',
                    'G': f'{grad_norm:.2f}'
                })

            end_time = time.time()
            duration = end_time - start_time
            print(f"\nTraining completed in {duration:.2f} seconds ({duration/60:.2f} minutes).")
            
            self._save_results()
    
    def _save_king_trades(self, king_num: int, formula: list, score: float, sharpe: float, ret: float):
        """保存 New King 的交易细节"""
        from .vm import StackVM
        
        output_dir = os.path.dirname(os.path.abspath(__file__))
        trades_dirs = [os.path.join(output_dir, 'king_trades')]
        train_dir = self.run_context.get("train_dir")
        if train_dir:
            trades_dirs.append(os.path.join(train_dir, 'king_trades'))
        for trades_dir in trades_dirs:
            os.makedirs(trades_dir, exist_ok=True)
        
        # 閲嶆柊鎵ц鍏紡鑾峰彇鍥犲瓙鍊?
        vm = StackVM()
        factors = vm.execute(formula, self.loader.feat_tensor, cs_mask=self.loader.cs_mask)
        
        if factors is None:
            return
        
        # 浣跨敤璇︾粏鍥炴祴鑾峰彇浜ゆ槗璁板綍
        bt = CBBacktest(top_k=RobustConfig.TOP_K)
        # 构造 TP 价格数据（与训练 worker 使用相同的 roll 对齐逻辑）
        king_open = None
        king_high = None
        king_prev_close = None
        if RobustConfig.TAKE_PROFIT > 0:
            if 'OPEN' in self.loader.raw_data_cache and 'HIGH' in self.loader.raw_data_cache:
                import torch as _torch
                raw_open = self.loader.raw_data_cache['OPEN']
                raw_high = self.loader.raw_data_cache['HIGH']
                close = self.loader.raw_data_cache['CLOSE']
                king_open = _torch.roll(raw_open, -1, dims=0)
                king_high = _torch.roll(raw_high, -1, dims=0)
                king_open[-1] = 1e9
                king_high[-1] = 1e9
                king_prev_close = close.clone()

        details = bt.evaluate_with_details(
            factors=factors,
            target_ret=self.loader.target_ret,
            valid_mask=self.loader.valid_mask,
            open_prices=king_open,
            high_prices=king_high,
            prev_close=king_prev_close,
        )
        
        # 鏋勫缓浜ゆ槗璁板綍
        trades = []
        for t, (indices, daily_ret) in enumerate(zip(details['daily_holdings'], details['daily_returns'])):
            date = self.loader.dates_list[t] if t < len(self.loader.dates_list) else f"Day_{t}"
            
            # 灏嗚祫浜х储寮曡浆鎹负鍚嶇О
            holdings = []
            for idx in indices:
                if idx < len(self.loader.assets_list):
                    code = self.loader.assets_list[idx]
                    name = self.loader.names_dict.get(code, code)
                    holdings.append(name)
            
            trades.append({
                'date': date,
                'holdings': holdings,
                'daily_ret': round(daily_ret, 6)
            })
        
        # 淇濆瓨鍒版枃浠?
        result = {
            'king_num': king_num,
            'formula': self.decode_formula(formula),
            'score': score,
            'sharpe': sharpe,
            'return': ret,
            'trades': trades
        }
        
        for trades_dir in trades_dirs:
            file_path = os.path.join(trades_dir, f'king_{king_num}.json')
            self._write_json(file_path, result)

    def _save_results(self):
        output_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 鏋勫缓瀹屾暣缁撴灉锛屽寘鍚渶浣冲洜瀛愬拰杩涘寲鍘嗗彶
        result = {
            'best': {
                'formula': self.best_formula,  # 鐜板湪鏄瓧绗︿覆鍒楄〃
                'readable': self.best_formula_readable,
                'raw_formula': self.best_formula_raw,
                'raw_readable': self.best_formula_raw_readable,
                'score': self.best_score,
                'sharpe': self.best_sharpe,
                'annualized_ret': self.best_return  # 骞村寲鏀剁泭鐜?
            },
            'history': self.king_history,
            'total_kings': len(self.king_history),
            'diverse_top_50': sorted(
                list(self.diverse_pool.values()),
                key=lambda x: x['score'],
                reverse=True
            )[:RobustConfig.DIVERSITY_POOL_SIZE]
        }
        
        result_path = os.path.join(output_dir, 'best_cb_formula.json')
        self._write_json(result_path, result)
        train_dir = self.run_context.get("train_dir")
        artifact_result_path = None
        if train_dir:
            artifact_result_path = os.path.join(train_dir, 'best_cb_formula.json')
            self._write_json(artifact_result_path, result)
        
        print(f"\nSaved to: {result_path}")
        print(f"   Total New Kings discovered: {len(self.king_history)}")
        print(f"   Best Score: {self.best_score:.2f}")
        print(f"   Best Sharpe: {self.best_sharpe:.2f}")
        print(f"   Best Annualized Return: {self.best_return:.2%}")
        print(f"   Best Formula: {self.best_formula_readable}")
        
        model_path = os.path.join(output_dir, 'alphagpt_cb.pt')
        torch.save(self.model.state_dict(), model_path)
        artifact_model_path = None
        if train_dir:
            artifact_model_path = os.path.join(train_dir, 'alphagpt_cb.pt')
            torch.save(self.model.state_dict(), artifact_model_path)

        if self.run_context:
            update_training_manifest(
                self.run_context,
                stage="train_completed",
                artifacts={
                    "best_formula_path": artifact_result_path or result_path,
                    "legacy_best_formula_path": result_path,
                    "model_weight_path": artifact_model_path or model_path,
                    "legacy_model_weight_path": model_path,
                    "king_trades_dir": os.path.join(train_dir, "king_trades") if train_dir else os.path.join(output_dir, "king_trades"),
                },
                summary={
                    "best_score": float(self.best_score),
                    "best_sharpe": float(self.best_sharpe),
                    "best_annualized_ret": float(self.best_return),
                    "best_formula_readable": self.best_formula_readable,
                    "total_kings": int(len(self.king_history)),
                    "data_start_date": self.data_start_date,
                },
            )

if __name__ == "__main__":
    # 鍛戒护琛屽弬鏁拌В鏋?
    parser = argparse.ArgumentParser(
        description='AlphaGPT training engine - convertible bond factor mining',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\\n"
            "  python -m model_core.engine\\n"
            "  python -m model_core.engine --config my_config.yaml"
        ),
    )
    parser.add_argument(
        '--config', '-c',
        type=str,
        default=None,
        help='閰嶇疆鏂囦欢璺緞 (YAML 鏍煎紡)锛屼笉鎸囧畾鍒欎娇鐢?default_config.yaml'
    )
    parser.add_argument(
        '--data-start-date',
        type=str,
        default=None,
        help='data start date (YYYY-MM-DD), default=2022-08-01'
    )
    parser.add_argument(
        '--run-id',
        type=str,
        default=None,
        help='optional training run id for artifacts/runs/<run_id>'
    )
    parser.add_argument(
        '--artifacts-root',
        type=str,
        default=None,
        help='optional artifacts root, default=artifacts/runs'
    )
    args = parser.parse_args()
    
    # 鍔犺浇閰嶇疆 (蹇呴』鍦ㄥ垱寤?AlphaEngine 涔嬪墠)
    from .config_loader import load_config
    config = load_config(args.config)
    
    # 璁板綍閰嶇疆鏂囦欢璺緞浠ヤ究鍦?init 涓墦鍗?
    # 璁板綍閰嶇疆璺緞 (浠呯敤浜庢墦鍗?璋冭瘯)
    RobustConfig._config_path = args.config if args.config else "default_config.yaml"  # type: ignore[attr-defined]
    
    if args.config:
        print(f"Loaded custom config: {args.config}")
    else:
        print("Using default config: default_config.yaml")

    run_context = prepare_training_run(
        config=config,
        config_path=args.config,
        data_start_date=args.data_start_date,
        run_id=args.run_id,
        artifacts_root=args.artifacts_root,
    )
    print(f"Prepared training manifest: {run_context['manifest_path']}")
    update_training_manifest(
        run_context,
        stage="train_running",
        summary={"data_start_date": args.data_start_date or "2022-08-01"},
    )
    
    # Windows 涓轰簡鏀寔 ProcessPool锛屽繀椤昏鏈夎繖涓?protect
    eng = AlphaEngine(data_start_date=args.data_start_date, run_context=run_context)
    eng.train()

