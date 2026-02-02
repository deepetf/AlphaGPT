import sys
sys.path.insert(0, r"c:\Trading\Projects\AlphaGPT")

from model_core.data_loader import CBDataLoader

loader = CBDataLoader()
loader.load_data()

print(f"Total dates: {len(loader.dates_list)}")
print(f"First date: {loader.dates_list[0]}")
print(f"Last date: {loader.dates_list[-1]}")

# Find 2024-01-02
target = "2024-01-02"
if target in loader.dates_list:
    idx = loader.dates_list.index(target)
    print(f"\n{target} is at index {idx}")
else:
    print(f"\n{target} not found")
