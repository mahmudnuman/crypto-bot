from data.store import load
from config import POPULAR_COINS, QUOTE_ASSET

print(f"{'Coin':<12} {'5m rows':>12} {'1h rows':>10} {'1d rows':>8}  Status")
print('-' * 58)
ok, fail = 0, 0
for coin in POPULAR_COINS:
    sym = f"{coin}{QUOTE_ASSET}"
    try:
        r5  = len(load(sym, "5m"))
        r1h = len(load(sym, "1h"))
        r1d = len(load(sym, "1d"))
        status = "OK" if r5 >= 100000 and r1h >= 10000 else "PARTIAL"
        if status == "OK": ok += 1
        else: fail += 1
        print(f"{coin:<12} {r5:>12,} {r1h:>10,} {r1d:>8,}  {status}")
    except Exception as e:
        fail += 1
        print(f"{coin:<12} ERROR: {e}")
print('-' * 58)
print(f"Result: {ok}/25 coins ready   {fail} need attention")
