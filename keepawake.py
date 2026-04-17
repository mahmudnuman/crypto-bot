"""
keepawake.py - Prevents Windows from sleeping during training.
Simulates activity every 3 minutes using SetThreadExecutionState API.
Run in background: python keepawake.py
"""
import ctypes, time, sys
from datetime import datetime

# Windows API constants
ES_CONTINUOUS       = 0x80000000
ES_SYSTEM_REQUIRED  = 0x00000001
ES_DISPLAY_REQUIRED = 0x00000002

def keep_awake():
    print("=" * 45)
    print("  KeepAwake — Training Sleep Guard")
    print("=" * 45)
    print("  PC will NOT sleep until you stop this.")
    print("  Press Ctrl+C to restore normal sleep.")
    print("=" * 45)

    # Tell Windows: system + display required (no sleep)
    ctypes.windll.kernel32.SetThreadExecutionState(
        ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED
    )
    print(f"\n  [OK] Sleep prevention ACTIVE at {datetime.now().strftime('%H:%M:%S')}\n")

    try:
        while True:
            # Re-assert every 3 minutes (belt + suspenders)
            ctypes.windll.kernel32.SetThreadExecutionState(
                ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED
            )
            print(f"  [{datetime.now().strftime('%H:%M')}] Still awake — training running...", flush=True)
            time.sleep(180)  # 3 minutes
    except KeyboardInterrupt:
        # Restore normal sleep behavior on exit
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
        print("\n  [Restored] Normal sleep settings back. Goodbye!")

if __name__ == "__main__":
    keep_awake()
