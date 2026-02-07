
import pandas as pd
import os

data_path = r"c:\Trading\Projects\AlphaGPT\data\cb_data.pq"
if not os.path.exists(data_path):
    print(f"Data file not found: {data_path}")
    exit(1)

try:
    df = pd.read_parquet(data_path)
    print("Columns:", df.columns.tolist())
    print("Index:", df.index.names)
    
    # Try to find date column
    date_col = 'trade_date'
    if date_col in df.columns:
        dates = pd.to_datetime(df[date_col])
        print(f"Date Range ({date_col}): {dates.min()} to {dates.max()}")
    elif 'date' in df.columns:
        dates = pd.to_datetime(df['date'])
        print(f"Date Range (date): {dates.min()} to {dates.max()}")
    elif isinstance(df.index, pd.DatetimeIndex):
        print(f"Date Range (Index): {df.index.min()} to {df.index.max()}")
    else:
        # Check if index level is date
        try:
            dates = df.index.get_level_values('date')
            print(f"Date Range (Index Level): {dates.min()} to {dates.max()}")
        except:
             print("Could not find date info.")
             print(df.head())
    
    print(f"Total Rows: {len(df)}")

except Exception as e:
    print(f"Error reading parquet: {e}")
