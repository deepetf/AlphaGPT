
import json
import os

filepath = r'c:\Trading\Projects\AlphaGPT\model_core\training_stats.jsonl'
output_path = r'c:\Trading\Projects\AlphaGPT\stats_result_utf8.txt'

lines = []
with open(filepath, 'rb') as f:
    try:
        f.seek(-4096, 2)
    except:
        pass
    lines = f.read().decode('utf-8', 'ignore').splitlines()
    
last_line = lines[-1]
try:
    data = json.loads(last_line)
    
    with open(output_path, 'w', encoding='utf-8') as out:
        out.write(f"Step: {data.get('step')}\n")
        out.write("Stats:\n")
        out.write(json.dumps(data.get('stats'), indent=2))
        
    print("Done writing to", output_path)

except Exception as e:
    print("Error parsing JSON:", e)
    print("Last line sample:", last_line[:100])
