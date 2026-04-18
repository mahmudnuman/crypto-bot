"""
status_live.py - Prints live training status every 2 minutes in the terminal.
Like SSH monitoring. Run: python status_live.py
"""
import time, json, subprocess, os
from pathlib import Path
from datetime import datetime

CKPT_DIR    = Path("cache/checkpoints")
LOG_FILE    = Path("cache/run_all_log.txt")
REFRESH     = 120  # 2 minutes
COINS_TOTAL = 25


def gpu():
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,temperature.gpu",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=3)
        p = r.stdout.strip().split(",")
        if len(p) == 3:
            return f"{p[0].strip()} util | {p[1].strip()} VRAM | {p[2].strip()}°C"
    except: pass
    return "GPU info unavailable"


def last_log_lines(n=3):
    if not LOG_FILE.exists(): return ["(no log yet)"]
    try: return LOG_FILE.read_text(errors="replace").splitlines()[-n:]
    except: return []


def summary():
    if not CKPT_DIR.exists(): return 0, 0, None, []
    done, in_prog, current, results = 0, 0, None, []
    for f in CKPT_DIR.glob("*.json"):
        if "fold" in f.stem: continue
        try:
            d = json.loads(f.read_text())
            status = d.get("status")
            best   = d.get("best_acc", 0)
            folds  = len(d.get("folds", []))
            if status == "done":
                done += 1
                results.append((f.stem, best, folds, "DONE"))
            elif status == "in_progress":
                in_prog += 1
                current = (f.stem, best, folds)
                results.append((f.stem, best, folds, "IN_PROG"))
        except: pass
    return done, in_prog, current, results


def print_update(tick):
    os.system("cls" if os.name == "nt" else "clear")
    now    = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    done, in_prog, current, results = summary()
    pct    = done / COINS_TOTAL * 100
    bar    = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))

    print(f"{'='*60}")
    print(f"  CryptoPredictBot — Live Status  [update #{tick}]")
    print(f"  {now}")
    print(f"{'='*60}")
    print(f"  GPU   : {gpu()}")
    print(f"  Coins : {done}/{COINS_TOTAL}  [{bar}]  {pct:.0f}%")
    print()

    if current:
        sym, best, folds = current
        print(f"  ▶ TRAINING NOW: {sym}")
        print(f"    Folds done: {folds}/10   Best acc: {best:.4f}")
    else:
        print("  ▶ No coin currently training")
    print()

    # Completed coins
    done_list = [(s,b,f) for s,b,f,st in results if st=="DONE"]
    if done_list:
        print(f"  ✓ COMPLETED ({len(done_list)} coins):")
        for sym, best, folds in sorted(done_list, key=lambda x: x[1], reverse=True):
            grade = "A★★★" if best>=0.62 else "B★★" if best>=0.57 else "C★" if best>=0.53 else "D"
            print(f"    {sym:<14} acc={best:.4f}  folds={folds}  {grade}")
    print()

    # Last log lines
    print("  Recent activity:")
    for line in last_log_lines(4):
        print(f"    {line}")

    print()
    print(f"  [Updates every {REFRESH//60} min — Ctrl+C to stop]")


def main():
    tick = 0
    print("  Starting live monitor (2-min updates)...")
    while True:
        try:
            tick += 1
            print_update(tick)
            time.sleep(REFRESH)
        except KeyboardInterrupt:
            print("\n  Stopped.")
            break


if __name__ == "__main__":
    main()
