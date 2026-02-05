import pandas as pd
try:
    path = r"C:\Trading\Projects\AlphaGPT\data\cb_data.pq"
    df = pd.read_parquet(path)
    with open('cols.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(df.columns.tolist()))
    print("Columns written to cols.txt")
except Exception as e:
    with open('cols.txt', 'w', encoding='utf-8') as f:
        f.write(f"Error: {e}")
