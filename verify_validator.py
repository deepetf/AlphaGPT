
import sys
import os
import torch

# Ensure model_core can be imported
sys.path.append(os.getcwd())

from model_core.config_loader import load_config
from model_core.formula_validator import validate_formula
from model_core.config import ModelConfig

def run_test():
    # 1. Load Config (simulating engine startup)
    config_path = "model_core/test_config.yaml"
    print(f"Loading config from {config_path}...")
    load_config(config_path)
    
    # 2. Check Input Features
    features = ModelConfig.INPUT_FEATURES
    print(f"Loaded {len(features)} features: {features[:5]}...")
    
    # 3. Test Formula
    formula = [
      "VOLATILITY_STK",
      "PCT_CHG_5",
      "LOG",
      "IF_POS",
      "ALPHA_PCT_CHG_5",
      "DIV",
      "CUT_NEG",
      "SQRT",
      "PREM",
      "LOG",
      "MUL",
      "TS_STD5"
    ]
    print(f"\nTesting Formula: {formula}")
    
    # 4. Run Validator
    is_valid, penalty, reason = validate_formula(formula)
    
    print("\nResult:")
    print(f"Valid: {is_valid}")
    print(f"Penalty: {penalty}")
    print(f"Reason: {reason}")
    
    if not is_valid:
        print("\nDEBUG INFO:")
        print("Checking tokens against features:")
        from model_core.formula_validator import _get_validation_cache
        ops, feats = _get_validation_cache()
        for token in formula:
            if token in feats:
                print(f"  {token}: FEATURE")
            elif token in ops:
                print(f"  {token}: OP (Arity={ops[token]})")
            else:
                print(f"  {token}: UNKNOWN <<<<")

if __name__ == "__main__":
    run_test()
