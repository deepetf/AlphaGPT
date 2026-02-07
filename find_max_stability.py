import json
import sys

# Set encoding for output
sys.stdout.reconfigure(encoding='utf-8')

try:
    with open(r'c:\\Trading\\Projects\\AlphaGPT\\model_core\\best_cb_formula.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    items = data.get('history', []) + data.get('diverse_top_50', [])
    # Filter out items without stability
    items = [i for i in items if 'stability' in i]
    
    if not items:
        print("No stability data found.")
    else:
        best = max(items, key=lambda x: x['stability'])
        print(f"Max Stability: {best['stability']}")
        print(f"Step: {best['step']}")
        print(f"Readable: {best['readable']}")

except Exception as e:
    print(f"Error: {e}")
