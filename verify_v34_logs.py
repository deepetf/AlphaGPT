
import json
import os

filepath = r'c:\Trading\Projects\AlphaGPT\model_core\training_stats.jsonl'
output_path = r'c:\Trading\Projects\AlphaGPT\v34_analysis.txt'

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
    
    with open(output_path, 'w', encoding='utf-8') as out:
        out.write(f"Total Runs: {len(runs)}\n")
        if latest_run:
            start_step = latest_run[0]['step']
            end_step = latest_run[-1]['step']
            count = len(latest_run)
            out.write(f"Latest Run (V3.4 Verification): Steps {start_step} -> {end_step} ({count} steps)\n\n")
            
            # Aggregate stats for the whole run
            total_stats = {}
            for row in latest_run:
                stats = row.get('stats', {})
                for k, v in stats.items():
                    total_stats[k] = total_stats.get(k, 0) + v
            
            out.write("Aggregated Stats for Latest Run:\n")
            out.write(json.dumps(total_stats, indent=2))
            out.write("\n\n")
            
            total_samples = sum(total_stats.values())
            out.write(f"STRUCT_INVALID Rate: {total_stats.get('STRUCT_INVALID', 0) / (total_samples if total_samples else 1) * 100:.2f}%\n")
            out.write(f"PASS Count: {total_stats.get('PASS', 0)}\n")
            
            out.write("\nLast 5 Steps Detail:\n")
            for row in latest_run[-5:]:
                out.write(f"Step {row['step']}: {json.dumps(row['stats'])}\n")

    print(f"Latest run stats exported to {output_path}")

except Exception as e:
    print(f"Error: {e}")
