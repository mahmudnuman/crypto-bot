"""
watchdog.py - Keeps run_all.py alive. If it crashes or stops, restarts it.
Run this INSTEAD of run_all.py: python watchdog.py
"""
import subprocess, time, sys, os
from pathlib import Path
from datetime import datetime

REPO_DIR   = Path(__file__).parent
RESTART_WAIT = 30  # seconds before restarting after a crash

def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"  [{ts}] {msg}", flush=True)

def main():
    print("=" * 50)
    print("  Watchdog — Auto-restart Training Guard")
    print("  Runs run_all.py and restarts if it stops.")
    print("  Press Ctrl+C to stop everything.")
    print("=" * 50)

    run_count = 0
    while True:
        run_count += 1
        log(f"Starting run_all.py (attempt #{run_count})...")

        try:
            proc = subprocess.Popen(
                [sys.executable, "run_all.py"],
                cwd=str(REPO_DIR),
                # Inherit stdout/stderr so you see output in this window
            )
            proc.wait()  # Block until it exits

            if proc.returncode == 0:
                log("Training COMPLETED successfully! All 25 coins done.")
                break  # Don't restart — it finished normally

            log(f"run_all.py exited with code {proc.returncode}. Restarting in {RESTART_WAIT}s...")

        except KeyboardInterrupt:
            log("Watchdog stopped by user.")
            try: proc.terminate()
            except: pass
            break
        except Exception as e:
            log(f"Error: {e}. Restarting in {RESTART_WAIT}s...")

        time.sleep(RESTART_WAIT)

if __name__ == "__main__":
    main()
