
import json
import os

filepath = r'c:\Trading\Projects\AlphaGPT\model_core\training_stats.jsonl'
output_path = r'c:\Trading\Projects\AlphaGPT\latest_stats_analysis.txt'

try:
    with open(filepath, 'rb') as f:
        # Seek to end and read last 8KB to ensure we capture multiple lines incase of long lines
        try:
            f.seek(-8192, 2)
        except:
            f.seek(0)
            
        lines = f.read().decode('utf-8', 'ignore').splitlines()
        
    # Get last 5 valid JSON lines
    valid_stats = []
    for line in reversed(lines):
        if not line.strip(): continue
        try:
            data = json.loads(line)
            valid_stats.append(data)
            if len(valid_stats) >= 5:
                break
        except:
            continue
            
    with open(output_path, 'w', encoding='utf-8') as out:
        if not valid_stats:
            out.write("No valid stats found.")
        else:
            # Write in chronological order (oldest to newest)
            for data in reversed(valid_stats):
                out.write(f"Step: {data.get('step')}\n")
                out.write("Stats:\n")
                out.write(json.dumps(data.get('stats'), indent=2))
                out.write("\n" + "="*40 + "\n")

    print(f"Successfully analyzed {len(valid_stats)} steps.")

except Exception as e:
    print(f"Error: {e}")
