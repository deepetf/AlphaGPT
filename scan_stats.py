
import json
import os

filepath = r'c:\Trading\Projects\AlphaGPT\model_core\training_stats.jsonl'
output_path = r'c:\Trading\Projects\AlphaGPT\stats_scan.txt'

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
                    # New run detected
                    if current_run:
                        runs.append(current_run)
                    current_run = []
                
                current_run.append(data)
                last_step = step
            except:
                continue
    
    if current_run:
        runs.append(current_run)
        
    with open(output_path, 'w', encoding='utf-8') as out:
        out.write(f"Total Runs Detected: {len(runs)}\n")
        
        for i, run in enumerate(runs):
            start_step = run[0]['step']
            end_step = run[-1]['step']
            count = len(run)
            out.write(f"Run {i+1}: Steps {start_step} -> {end_step} (Rows: {count})\n")
            
        out.write("\n=== Latest Run Last 5 Steps ===\n")
        if runs:
            latest = runs[-1][-5:]
            for data in latest:
                out.write(f"Step: {data.get('step')}\n")
                out.write(json.dumps(data.get('stats'), indent=2))
                out.write("\n" + "-"*20 + "\n")

    print(f"Scanned {len(runs)} runs.")

except Exception as e:
    print(f"Error: {e}")
