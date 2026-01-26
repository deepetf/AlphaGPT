
import pandas as pd
import os

PARQUET_PATH = r"C:\Trading\Projects\AlphaGPT\data\cb_data.pq"

def inspect():
    if not os.path.exists(PARQUET_PATH):
        print(f"File not found: {PARQUET_PATH}")
        return

    print(f"Reading {PARQUET_PATH}...")
    try:
        df = pd.read_parquet(PARQUET_PATH)
    except Exception as e:
        print(f"Error reading parquet: {e}")
        return

    print("\n" + "="*50)
    print("📊 DATA INSPECTION REPORT")
    print("="*50)
    
    print(f"\n1. Shape: {df.shape}")
    
    print(f"\n2. Index Names: {df.index.names}")
    
    print("\n3. Index Sample:")
    print(df.index[:5])
    
    print("\n4. Columns:")
    print(df.columns.tolist())
    
    print("\n5. Dtypes:")
    print(df.dtypes)
    
    print("\n6. Head (First 3 rows):")
    print(df.head(3))
    
    # 检查关键列
    required_cols = ['trade_date', 'code', 'close', 'vol']
    missing = []
    
    # Check if index has required levels
    if 'trade_date' not in df.columns and 'trade_date' not in df.index.names:
        missing.append('trade_date')
    if 'code' not in df.columns and 'code' not in df.index.names:
        missing.append('code')
        
    for col in ['close', 'vol']:
        if col not in df.columns:
            missing.append(col)
            
    if missing:
        print(f"\n❌ MISSING CRITICAL COLUMNS/INDEX: {missing}")
    else:
        print("\n✅ Critical columns/index present.")
        
    # 检查新请求的列
    print("\n7. Checking requested new factors:")
    targets = ['IV', 'stock_vol60d', 'convprem_zscore']
    cols_set = set(df.columns)
    for t in targets:
        if t in cols_set:
             print(f"   ✅ '{t}' found.")
        else:
             print(f"   ❌ '{t}' NOT found.")
             # Fuzzy match
             similar = [c for c in df.columns if t.lower() in c.lower() or c.lower() in t.lower()]
             if similar:
                 print(f"      Did you mean: {similar}?")

if __name__ == "__main__":
    inspect()
