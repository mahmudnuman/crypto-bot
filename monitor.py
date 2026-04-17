"""
monitor.py — Live training progress dashboard.
Run in a separate terminal: python monitor.py
Auto-refreshes every 5 seconds.
"""
import sys, time, re, os
from pathlib import Path
from datetime import datetime

# ── Try rich for beautiful UI, fall back to plain ────────────────────
try:
    from rich.console import Console
    from rich.table import Table
    from rich.live import Live
    from rich.panel import Panel
    from rich.columns import Columns
    from rich.text import Text
    from rich import box
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

sys.path.insert(0, str(Path(__file__).parent))

LOG_FILE    = Path("cache/run_all_log.txt")
MODELS_DIR  = Path("saved_models")
REPORT_FILE = Path("cache/training_report.json")
COINS_TOTAL = 25

# ── Parse log file ────────────────────────────────────────────────────
def parse_log():
    if not LOG_FILE.exists():
        return {"folds": [], "coins_done": [], "coins_failed": [], "current": None, "started": None}

    text = LOG_FILE.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()

    folds        = []   # (coin, fold, train_acc, test_acc, mape)
    coins_done   = []   # (coin, acc, elapsed_str)
    coins_failed = []
    current_coin = None
    started      = None

    for line in lines:
        # Started
        if "run_all.py started" in line:
            m = re.match(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", line)
            if m:
                started = m.group(1)

        # Current coin training
        m = re.search(r"\[(\d+)/25\] Training (\w+)", line)
        if m:
            current_coin = m.group(2)

        # Fold result  — "  [BTCUSDT] Fold 00 train=0.564 test=0.521 mape=0.015"
        m = re.search(r"\[(\w+)\] Fold (\d+) train=([\d.]+) test=([\d.]+) mape=([\d.nan]+)", line)
        if m:
            folds.append({
                "coin":  m.group(1),
                "fold":  int(m.group(2)),
                "train": float(m.group(3)),
                "test":  float(m.group(4)),
                "mape":  m.group(5),
            })

        # Coin done  — "BTCUSDT DONE: acc=0.521 in 23.4min"
        m = re.search(r"(\w+USDT) DONE: acc=([\d.]+) in ([\d.]+)min", line)
        if m:
            coins_done.append({"coin": m.group(1), "acc": float(m.group(2)), "min": m.group(3)})

        # Failed
        if "failed:" in line.lower() and "fold" not in line.lower():
            m = re.search(r"(\w+USDT)", line)
            if m and m.group(1) not in coins_failed:
                coins_failed.append(m.group(1))

    return {
        "folds":        folds,
        "coins_done":   coins_done,
        "coins_failed": coins_failed,
        "current":      current_coin,
        "started":      started,
    }


def saved_model_count():
    if not MODELS_DIR.exists():
        return 0
    return len(list(MODELS_DIR.glob("*_ensemble.pkl")))


def gpu_util():
    try:
        import subprocess
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=3
        )
        parts = r.stdout.strip().split(",")
        if len(parts) == 3:
            return parts[0].strip(), parts[1].strip(), parts[2].strip()
    except Exception:
        pass
    return "?", "?", "?"


def elapsed_since(started_str):
    if not started_str:
        return "?"
    try:
        t = datetime.strptime(started_str, "%Y-%m-%d %H:%M:%S")
        secs = (datetime.now() - t).total_seconds()
        h, m = int(secs // 3600), int((secs % 3600) // 60)
        return f"{h}h {m:02d}m"
    except Exception:
        return "?"


# ── Rich dashboard ─────────────────────────────────────────────────────
def make_dashboard(data, models_saved):
    console = Console()
    gpu_pct, gpu_mem, gpu_tot = gpu_util()
    elapsed   = elapsed_since(data["started"])
    folds     = data["folds"]
    done      = data["coins_done"]
    current   = data["current"] or "—"
    n_folds   = len(folds)
    n_done    = models_saved
    now_str   = datetime.now().strftime("%H:%M:%S")

    # ── Header ────────────────────────────────────────────────────────
    header = Panel(
        f"[bold cyan] CryptoPredictBot — Training Monitor[/bold cyan]   "
        f"[dim]Updated: {now_str}[/dim]",
        style="bold blue", box=box.DOUBLE_EDGE
    )

    # ── Summary row ───────────────────────────────────────────────────
    pct = (n_done / COINS_TOTAL) * 100
    bar_filled = int(pct / 5)
    bar = "[green]" + "█" * bar_filled + "[/green]" + "░" * (20 - bar_filled)
    summary = Panel(
        f"[bold]Coins done:[/bold]  {n_done}/{COINS_TOTAL}  {bar}  {pct:.0f}%\n"
        f"[bold]Current:[/bold]     [yellow]{current}[/yellow]\n"
        f"[bold]Folds done:[/bold]  {n_folds}\n"
        f"[bold]Elapsed:[/bold]     {elapsed}",
        title="[bold]Progress[/bold]", box=box.ROUNDED
    )

    gpu_color = "green" if gpu_pct != "?" and int(gpu_pct.replace("%","").strip() or 0) > 20 else "red"
    gpu_panel = Panel(
        f"[bold]GPU (NVIDIA MX130)[/bold]\n"
        f"Utilization: [{gpu_color}]{gpu_pct}[/{gpu_color}]\n"
        f"VRAM:        {gpu_mem} / {gpu_tot}",
        title="[bold]Hardware[/bold]", box=box.ROUNDED
    )

    # ── Recent folds table ────────────────────────────────────────────
    fold_table = Table(title="Recent Folds", box=box.SIMPLE_HEAD, show_lines=False)
    fold_table.add_column("Coin",      style="cyan",  width=12)
    fold_table.add_column("Fold",      justify="right", width=6)
    fold_table.add_column("Train Acc", justify="right", width=10)
    fold_table.add_column("Test Acc",  justify="right", width=10)
    fold_table.add_column("MAPE",      justify="right", width=8)
    fold_table.add_column("Signal",    width=10)

    for f in folds[-15:]:
        acc = f["test"]
        if acc >= 0.57:    sig, col = "STRONG ↑", "green"
        elif acc >= 0.53:  sig, col = "OK",        "yellow"
        else:              sig, col = "weak",       "dim"
        fold_table.add_row(
            f["coin"], str(f["fold"]),
            f"{f['train']:.3f}", f"[{col}]{acc:.3f}[/{col}]",
            f["mape"], f"[{col}]{sig}[/{col}]"
        )

    # ── Completed coins table ─────────────────────────────────────────
    done_table = Table(title=f"Completed Coins ({n_done}/{COINS_TOTAL})",
                       box=box.SIMPLE_HEAD, show_lines=False)
    done_table.add_column("Coin",     style="cyan", width=12)
    done_table.add_column("Best Acc", justify="right", width=10)
    done_table.add_column("Time",     justify="right", width=8)
    done_table.add_column("Grade",    width=8)

    for d in reversed(done):
        acc = d["acc"]
        if acc >= 0.60:   grade, col = "A ★★★", "green"
        elif acc >= 0.55: grade, col = "B ★★",  "yellow"
        elif acc >= 0.52: grade, col = "C ★",   "white"
        else:             grade, col = "D",      "dim"
        done_table.add_row(
            d["coin"], f"[{col}]{acc:.3f}[/{col}]",
            f"{d['min']}m", f"[{col}]{grade}[/{col}]"
        )
    if not done:
        done_table.add_row("[dim]—[/dim]", "[dim]training...[/dim]", "", "")

    return header, summary, gpu_panel, fold_table, done_table


# ── Plain text fallback ────────────────────────────────────────────────
def plain_refresh():
    os.system("cls" if os.name == "nt" else "clear")
    data = parse_log()
    n_done = saved_model_count()
    gpu_pct, gpu_mem, gpu_tot = gpu_util()
    now = datetime.now().strftime("%H:%M:%S")

    print(f"{'='*55}")
    print(f"  CryptoPredictBot — Training Monitor   [{now}]")
    print(f"{'='*55}")
    print(f"  Coins done : {n_done}/{COINS_TOTAL}")
    print(f"  Current    : {data['current'] or '—'}")
    print(f"  Folds done : {len(data['folds'])}")
    print(f"  GPU        : {gpu_pct}  VRAM: {gpu_mem}/{gpu_tot}")
    print(f"  Elapsed    : {elapsed_since(data['started'])}")
    print()

    if data["folds"]:
        print(f"  {'Coin':<12} {'Fold':>4} {'Train':>7} {'Test':>7} {'MAPE':>7}")
        print(f"  {'-'*42}")
        for f in data["folds"][-12:]:
            sig = "▲" if f["test"] >= 0.53 else "▽"
            print(f"  {f['coin']:<12} {f['fold']:>4}  {f['train']:>6.3f}  {f['test']:>6.3f} {sig}  {f['mape']:>6}")
    print()

    if data["coins_done"]:
        print(f"  Completed coins:")
        for d in data["coins_done"]:
            print(f"    ✓ {d['coin']:<14} acc={d['acc']:.3f}  ({d['min']}min)")
    print()
    print("  [Ctrl+C to exit]")


# ── Main loop ─────────────────────────────────────────────────────────
def main():
    refresh = 5  # seconds

    if not HAS_RICH:
        print("(rich not installed — using plain mode)")
        while True:
            try:
                plain_refresh()
                time.sleep(refresh)
            except KeyboardInterrupt:
                print("\nMonitor stopped.")
                break
        return

    console = Console()

    with Live(console=console, refresh_per_second=0.2, screen=True) as live:
        while True:
            try:
                data   = parse_log()
                n_done = saved_model_count()
                header, summary, gpu_panel, fold_table, done_table = make_dashboard(data, n_done)

                from rich.layout import Layout
                from rich.columns import Columns

                layout = Layout()
                layout.split_column(
                    Layout(header,   size=3),
                    Layout(Columns([summary, gpu_panel]), size=8),
                    Layout(fold_table,  ratio=2),
                    Layout(done_table,  ratio=1),
                )
                live.update(layout)
                time.sleep(refresh)

            except KeyboardInterrupt:
                console.print("\n[yellow]Monitor stopped.[/yellow]")
                break
            except Exception as e:
                # Fall back to simple print on any rich error
                console.print(f"[red]Render error: {e}[/red]")
                time.sleep(refresh)


if __name__ == "__main__":
    main()
