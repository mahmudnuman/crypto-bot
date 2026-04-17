"""
autopush.py — Automatically commits and pushes progress to GitHub.
Runs every 10 minutes in a separate terminal window.

Usage: python autopush.py
"""
import subprocess, time, sys, os
from pathlib import Path
from datetime import datetime

REPO_DIR     = Path(__file__).parent
PUSH_EVERY_S = 600  # 10 minutes

def run(cmd, cwd=None):
    r = subprocess.run(cmd, capture_output=True, text=True,
                       cwd=cwd or str(REPO_DIR), shell=True)
    return r.stdout.strip(), r.stderr.strip(), r.returncode

def count_trained():
    ckpt_dir = REPO_DIR / "cache" / "checkpoints"
    if not ckpt_dir.exists():
        return 0, 0
    import json
    done = 0
    folds_total = 0
    for f in ckpt_dir.glob("*.json"):
        try:
            d = json.loads(f.read_text())
            if d.get("status") == "done":
                done += 1
            folds_total += len(d.get("folds", []))
        except:
            pass
    return done, folds_total

def git_push():
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    done, folds = count_trained()

    # Check if there's anything to commit
    out, _, _ = run("git status --porcelain")
    if not out.strip():
        print(f"  [{datetime.now().strftime('%H:%M')}] Nothing new to commit.")
        return False

    msg = f"training update {now} — {done}/25 coins done, {folds} folds"

    run("git add -A")
    out, err, code = run(f'git commit -m "{msg}"')
    if code != 0:
        print(f"  Commit failed: {err[:100]}")
        return False

    out, err, code = run("git push origin main")
    if code != 0:
        # Try master branch
        out, err, code = run("git push origin master")
    if code != 0:
        print(f"  Push failed: {err[:150]}")
        return False

    print(f"  [{datetime.now().strftime('%H:%M')}] ✓ Pushed — {msg}")
    return True

def main():
    print("=" * 50)
    print("  AutoPush — GitHub Sync Every 10 min")
    print("=" * 50)

    # Test git is configured
    out, err, code = run("git remote -v")
    if code != 0 or not out:
        print("\n  ERROR: No git remote configured.")
        print("  Run this first:\n")
        print("    git remote add origin https://github.com/YOUR_USERNAME/crypto-bot.git")
        print("\n  Then restart this script.")
        sys.exit(1)

    print(f"\n  Remote: {out.splitlines()[0]}")
    print(f"  Pushing every {PUSH_EVERY_S//60} minutes.")
    print("  Press Ctrl+C to stop.\n")

    push_count = 0
    while True:
        try:
            result = git_push()
            if result:
                push_count += 1
            next_push = datetime.fromtimestamp(
                time.time() + PUSH_EVERY_S).strftime("%H:%M")
            print(f"  Next push at {next_push} (push #{push_count+1})")
            time.sleep(PUSH_EVERY_S)
        except KeyboardInterrupt:
            print("\n  AutoPush stopped.")
            break
        except Exception as e:
            print(f"  Error: {e}")
            time.sleep(60)

if __name__ == "__main__":
    main()
