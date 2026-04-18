import pandas as pd
from datetime import timedelta

# Check feature cache
df = pd.read_parquet('cache/features/MATICUSDT_features.parquet')
print(f"Feature cache: {len(df):,} rows")
print(f"  Date range:  {df.index.min()} -> {df.index.max()}")
print(f"  Total days:  {(df.index.max() - df.index.min()).days}")

from config import WFV_INITIAL_TRAIN_YEARS, WFV_GAP_DAYS, WFV_TEST_DAYS, WFV_STEP_DAYS
t_start = df.index.min()
t_end   = df.index.max()
first_train_end = t_start + timedelta(days=WFV_INITIAL_TRAIN_YEARS * 365)

print(f"\nWFV settings:")
print(f"  Initial train: {WFV_INITIAL_TRAIN_YEARS} years ({WFV_INITIAL_TRAIN_YEARS*365:.0f} days)")
print(f"  Step:          {WFV_STEP_DAYS} days")
print(f"  Gap:           {WFV_GAP_DAYS} days")
print(f"  Test window:   {WFV_TEST_DAYS} days")
print(f"\n  Train ends at:  {first_train_end}")
print(f"  Data ends at:   {t_end}")
print(f"  Remaining days: {(t_end - first_train_end).days}")

# Count folds
count = 0
scan = first_train_end
while True:
    ts = scan + timedelta(days=WFV_GAP_DAYS)
    te = ts   + timedelta(days=WFV_TEST_DAYS)
    if te > t_end: break
    count += 1
    scan += timedelta(days=WFV_STEP_DAYS)

print(f"\n  Total possible folds: {count}")
if count < 2:
    print("\n  *** DIAGNOSIS: MATIC listed on Binance recently — ")
    print("      not enough historical data for multiple WFV folds!")
    print(f"      After {WFV_INITIAL_TRAIN_YEARS}yr training, only {(t_end-first_train_end).days} days left.")
    print(f"      Need at least {WFV_GAP_DAYS+WFV_TEST_DAYS+WFV_STEP_DAYS}d per fold.")
