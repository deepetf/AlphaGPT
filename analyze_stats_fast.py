import json
import os
from collections import Counter

def analyze_log(file_path):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    if not data:
        print("No valid data found.")
        return

    total_counts = Counter()
    grand_total = 0
    
    # Store per-step totals to check trends
    step_metrics = []

    for entry in data:
        stats = entry.get('stats', {})
        step_total = sum(stats.values())
        grand_total += step_total
        total_counts.update(stats)
        
        step_metrics.append({
            'step': entry.get('step'),
            'total': step_total,
            'pass': stats.get('PASS', 0)
        })

    print("=== Training Efficiency Analysis (Fast) ===")
    print(f"Total Steps Analyzed: {len(data)}")
    print(f"Total Samples: {grand_total}")
    print("-" * 40)
    
    # Sort by count desc
    distribution = sorted(total_counts.items(), key=lambda x: x[1], reverse=True)
    
    print("Failure Reason Distribution:")
    for reason, count in distribution:
        pct = (count / grand_total * 100) if grand_total > 0 else 0
        print(f"  {reason:<20}: {pct:.2f}%")
        
    print("-" * 40)
    
    # PASS Rate Trend
    if step_metrics:
        mid = len(step_metrics) // 2
        first = step_metrics[:mid]
        last = step_metrics[mid:]
        
        def calc_rate(items):
            t = sum(x['total'] for x in items)
            p = sum(x['pass'] for x in items)
            return (p/t*100) if t > 0 else 0
            
        r1 = calc_rate(first)
        r2 = calc_rate(last)
        print(f"PASS Rate Trend: {r1:.2f}% -> {r2:.2f}%")

    # Grouped Analysis
    print("-" * 40)
    groups = {
        'Structure': ['STRUCT_INVALID'],
        'LowVariance': ['LOW_VARIANCE'],
        'Execution': ['EXEC_NONE', 'EXEC_ERR'],
        'Metrics': ['METRIC_SHARPE', 'METRIC_ACTIVE', 'METRIC_FLIP', 'METRIC_DAYS'],
        'Similarity': ['SIM_REJECT', 'SIM_REPLACE']
    }
    
    print("Grouped Analysis:")
    for group_name, prefixes in groups.items():
        # Match exact keys or prefixes
        group_sum = 0
        matched_keys = []
        for key, count in total_counts.items():
            if key in prefixes:
                group_sum += count
                matched_keys.append(key)
        
        if grand_total > 0:
            print(f"  {group_name:<15}: {group_sum/grand_total*100:.2f}%")
            if group_name == 'Metrics':
                for k in matched_keys:
                    print(f"    - {k:<15}: {total_counts[k]/grand_total*100:.2f}%")

if __name__ == "__main__":
    analyze_log(r"c:\Trading\Projects\AlphaGPT\model_core\training_stats.jsonl")
