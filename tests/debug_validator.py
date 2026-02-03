
from model_core.formula_validator import validate_formula
from model_core.config import RobustConfig

# Print current config
print(f"Config: MaxTS={RobustConfig.MAX_TS_IN_WINDOW}, Penalty={RobustConfig.DENSITY_PENALTY}")

test_cases = [
    # 1. Valid Simple Formula
    (["PURE_VALUE", "ABS", "TS_MEAN5"], "Valid Simple"),
    
    # 2. Valid Deep Formula (should pass with MaxTS=4)
    (["PURE_VALUE", "TS_MEAN5", "TS_MEAN5", "TS_MEAN5", "TS_MEAN5"], "Boundary Case (4 TS)"),
    
    # 3. Invalid Stacked Formula (should fail or penalty)
    (["PURE_VALUE", "TS_MEAN5", "TS_MEAN5", "TS_MEAN5", "TS_MEAN5", "TS_MEAN5"], "Invalid Stack (5 TS)"),
    
    # 4. Old King (should pass)
    (['PURE_VALUE', 'ABS', 'SQRT', 'LOG', 'ABS', 'TS_MEAN5', 'ABS', 'TS_MEAN5', 'TS_MEAN5', 'ABS', 'TS_MEAN5', 'TS_MEAN5'], "Old King")
]

for formula, name in test_cases:
    is_valid, penalty, reason = validate_formula(formula)
    print(f"[{name}] Valid: {is_valid}, Penalty: {penalty}, Reason: {reason}")
