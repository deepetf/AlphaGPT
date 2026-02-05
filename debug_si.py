
import json
import os

filepath = r'c:\Trading\Projects\AlphaGPT\model_core\training_stats.jsonl'

try:
    runs = []
    current_run = []
    last_step = -1
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            try:
                data = json.loads(line)
                step = data.get('step', -1)
                
                if step < last_step:
                    if current_run:
                        runs.append(current_run)
                    current_run = []
                
                current_run.append(data)
                last_step = step
            except:
                continue
    
    if current_run:
        runs.append(current_run)
        
    latest_run = runs[-1] if runs else []
    
    print(f"Latest Run: Steps {latest_run[0]['step']} to {latest_run[-1]['step']}")
    
    for row in latest_run:
        si = row.get('stats', {}).get('STRUCT_INVALID', 0)
        if si > 0:
            print(f"Step {row['step']}: STRUCT_INVALID = {si}")

except Exception as e:
    print(f"Error: {e}")
