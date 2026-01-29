
import torch
from model_core.ops_registry import OpsRegistry

# Test TS_BIAS5
print("Testing TS_BIAS5 operator...")

# Get the operator
op_info = OpsRegistry.get_op('TS_BIAS5')
if not op_info:
    print("Error: TS_BIAS5 not registered!")
    exit(1)

func = op_info['func']

# Create dummy data
# Time=10, Assets=2
x = torch.tensor([
    [10.0, 100.0],
    [11.0, 101.0],
    [12.0, 102.0],
    [13.0, 103.0],
    [14.0, 104.0],
    [15.0, 105.0],  # MA5 for A = (11+12+13+14+15)/5 = 13.0. Bias = (15-13)/13 = 2/13
    [16.0, 106.0],
    [17.0, 107.0],
    [18.0, 108.0],
    [19.0, 109.0]
], dtype=torch.float32)

# Expected calculation for index 5 (6th day)
# MA5 for Asset 0: (11+12+13+14+15)/5 = 13.0
# BIAS for Asset 0: (15 - 13) / 13 = 0.153846

result = func(x)

print(f"Input shape: {x.shape}")
print(f"Result shape: {result.shape}")

bias_val_asset0 = result[5, 0].item()
print(f"Bias[5,0] = {bias_val_asset0:.6f}")

expected = (15.0 - 13.0) / 13.0
print(f"Expected  = {expected:.6f}")

if abs(bias_val_asset0 - expected) < 1e-5:
    print("✅ Logic Verified")
else:
    print("❌ Verification Failed")
