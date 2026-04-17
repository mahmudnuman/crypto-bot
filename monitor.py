"""
monitor.py — Live training progress dashboard (reads checkpoint files).
Run in a separate terminal: python monitor.py
Auto-refreshes every 5 seconds.
"""
import sys, time, json, os, subprocess
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

CKPT_DIR    = Path("cache/checkpoints")
LOG_FILE    = Path("cache/run_all_log.txt")
MODELS_DIR  = Path("saved_models")
COINS_TOTAL = 25
REFRESH     = 5

try:
    from rich.console import Console
    from rich.table import Table
    from rich.live import Live
    from rich.panel import Panel
    from rich.layout import Layout
    from rich.columns import Columns
    from rich.text import Text
    from rich import box
    HAS_RICH = True
except ImportError:
    HAS_RICH = False


# ── Data readers ──────────────────────────────────────────────────────
def gpu_info():
    try:
        r = subprocess.run(
            ["nvidia-smi","--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=3)
        p = r.stdout.strip().split(",")
        if len(p) == 4:
            return p[0].strip(), p[1].strip(), p[2].strip(), p[3].strip()
    except: pass
    return "?", "?", "?", "?"


def read_all_checkpoints():
    """Return list of coin checkpoint dicts, sorted by order they appear in log."""
    if not CKPT_DIR.exists():
        return []
    coins = []
    for f in sorted(CKPT_DIR.glob("*.json")):
        try:
            d = json.loads(f.read_text())
            d["symbol"] = f.stem
            coins.append(d)
        except: pass
    return coins


def get_current_coin():
    """Read last [N/25] Training XXXUSDT from log."""
    if not LOG_FILE.exists(): return None, None
    try:
        lines = LOG_FILE.read_text(encoding="utf-8", errors="replace").splitlines()
        for line in reversed(lines):
            import re
            m = re.search(r"\[(\d+)/25\] Training (\w+)", line)
            if m:
                return m.group(2), int(m.group(1))
    except: pass
    return None, None


def get_started_time():
    if not LOG_FILE.exists(): return None
    try:
        import re
        text = LOG_FILE.read_text(encoding="utf-8", errors="replace")
        for line in text.splitlines():
            if "run_all.py started" in line:
                m = re.match(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", line)
                if m: started = m.group(1)
        return started
    except: return None


def elapsed(started_str):
    if not started_str: return "?"
    try:
        t = datetime.strptime(started_str, "%Y-%m-%d %H:%M:%S")
        s = (datetime.now() - t).total_seconds()
        h, m = int(s//3600), int((s%3600)//60)
        return f"{h}h {m:02d}m"
    except: return "?"


# ── Plain fallback ─────────────────────────────────────────────────────
def plain_render():
    os.system("cls" if os.name=="nt" else "clear")
    coins       = read_all_checkpoints()
    current, ci = get_current_coin()
    gpu_pct, gpu_mem, gpu_tot, gpu_temp = gpu_info()
    now = datetime.now().strftime("%H:%M:%S")

    n_done = sum(1 for c in coins if c.get("status")=="done")
    total_folds = sum(len(c.get("folds",[])) for c in coins)

    print(f"{'='*60}")
    print(f"  CryptoPredictBot — Live Monitor  [{now}]")
    print(f"{'='*60}")
    print(f"  Coins    : {n_done}/25 done  |  {len(coins)} in progress")
    print(f"  Folds    : {total_folds} completed total")
    print(f"  GPU      : {gpu_pct}  {gpu_mem}/{gpu_tot}  {gpu_temp}°C")
    print(f"  Elapsed  : {elapsed(get_started_time())}")
    print()

    # Current coin folds
    cur_data = next((c for c in coins if c.get("symbol")==current), None)
    if cur_data:
        folds = cur_data.get("folds", [])
        print(f"  ── {current} [{ci}/25] — {len(folds)} folds done ──")
        print(f"  {'Fold':>5} {'Train':>7} {'Test':>7} {'MAPE':>8}  Status")
        print(f"  {'-'*42}")
        for f in folds:
            sig = "▲" if f["test_acc"] >= 0.53 else "▽"
            print(f"  {f['fold']:>5}  {f['train_acc']:>6.3f}  {f['test_acc']:>6.3f}  {f['mape']:>7.4f}  {sig}")
        print()

    # Completed coins
    done_coins = [c for c in coins if c.get("status")=="done"]
    if done_coins:
        print(f"  ── Completed Coins ──")
        print(f"  {'Coin':<14} {'Folds':>6} {'BestAcc':>8} {'Grade':>6}")
        print(f"  {'-'*38}")
        for c in done_coins:
            acc = c.get("best_acc",0)
            grade = "A★★★" if acc>=0.60 else "B★★" if acc>=0.55 else "C★" if acc>=0.52 else "D"
            print(f"  {c['symbol']:<14} {len(c.get('folds',[]))*' ':>6} {acc:>8.3f} {grade:>6}")
    print(f"\n  [Ctrl+C to exit]  Refreshes every {REFRESH}s")


# ── Rich dashboard ─────────────────────────────────────────────────────
def build_rich_layout():
    coins       = read_all_checkpoints()
    current, ci = get_current_coin()
    gpu_pct, gpu_mem, gpu_tot, gpu_temp = gpu_info()
    started     = get_started_time()
    now_str     = datetime.now().strftime("%H:%M:%S")

    n_done      = sum(1 for c in coins if c.get("status")=="done")
    total_folds = sum(len(c.get("folds",[])) for c in coins)
    in_prog     = [c for c in coins if c.get("status")=="in_progress"]
    done_coins  = [c for c in coins if c.get("status")=="done"]

    # ── Header ────────────────────────────────────────────────────────
    header = Panel(
        f"[bold cyan]CryptoPredictBot — Live Training Monitor[/bold cyan]"
        f"   [dim]{now_str}[/dim]",
        style="bold blue", box=box.DOUBLE_EDGE, height=3
    )

    # ── Progress panel ────────────────────────────────────────────────
    pct = (n_done / COINS_TOTAL) * 100
    bar = "[green]" + "█"*int(pct/5) + "[/green]" + "░"*(20-int(pct/5))
    cur_sym   = current or "—"
    cur_folds = len(in_prog[0].get("folds",[])) if in_prog else 0
    progress  = Panel(
        f"[bold]Overall:[/bold]  {n_done}/{COINS_TOTAL} coins  {bar}  {pct:.0f}%\n"
        f"[bold]Current:[/bold]  [yellow]{cur_sym}[/yellow]"
        + (f"  [{ci}/25]  fold {cur_folds}/10" if ci else "") + "\n"
        f"[bold]Folds:[/bold]    {total_folds} completed total\n"
        f"[bold]Elapsed:[/bold]  {elapsed(started)}",
        title="[bold]Progress[/bold]", box=box.ROUNDED
    )

    # ── GPU panel ─────────────────────────────────────────────────────
    try: g = int(gpu_pct.replace("%","").strip())
    except: g = 0
    gcol = "green" if g>20 else "red"
    gpu_panel = Panel(
        f"[bold]NVIDIA MX130[/bold]\n"
        f"GPU:   [{gcol}]{gpu_pct}[/{gcol}]\n"
        f"VRAM:  {gpu_mem} / {gpu_tot}\n"
        f"Temp:  {gpu_temp}°C",
        title="[bold]Hardware[/bold]", box=box.ROUNDED
    )

    # ── Current coin fold table ───────────────────────────────────────
    cur_data  = in_prog[0] if in_prog else None
    fold_tbl  = Table(
        title=f"[bold yellow]{cur_sym}[/bold yellow] — Fold Results",
        box=box.SIMPLE_HEAD, show_lines=False, expand=True
    )
    fold_tbl.add_column("Fold",      justify="center", width=6)
    fold_tbl.add_column("Train Acc", justify="right",  width=10)
    fold_tbl.add_column("Test Acc",  justify="right",  width=10)
    fold_tbl.add_column("MAPE",      justify="right",  width=9)
    fold_tbl.add_column("Time",      justify="right",  width=8)
    fold_tbl.add_column("Status",    width=12)

    if cur_data:
        for f in cur_data.get("folds", []):
            acc  = f["test_acc"]
            secs = f.get("duration_s", 0)
            mins = f"{secs/60:.1f}m" if secs else "—"
            if acc >= 0.56:   sig, col = "STRONG ▲", "green"
            elif acc >= 0.53: sig, col = "Good ▲",   "yellow"
            elif acc >= 0.51: sig, col = "OK",        "white"
            else:             sig, col = "weak ▽",    "dim"
            fold_tbl.add_row(
                str(f["fold"]),
                f"{f['train_acc']:.4f}",
                f"[{col}]{acc:.4f}[/{col}]",
                f"{f['mape']:.4f}",
                f"[dim]{mins}[/dim]",
                f"[{col}]{sig}[/{col}]"
            )
        fold_tbl.add_row(
            "[dim]...[/dim]", "[dim]training[/dim]",
            "[dim]—[/dim]", "[dim]—[/dim]", "[dim]—[/dim]", "[dim]running...[/dim]"
        )
    else:
        fold_tbl.add_row("[dim]—[/dim]","[dim]waiting[/dim]","","","","")

    # ── Completed coins table ─────────────────────────────────────────
    done_tbl = Table(
        title=f"[bold green]Completed Coins ({n_done}/25)[/bold green]",
        box=box.SIMPLE_HEAD, show_lines=False, expand=True
    )
    done_tbl.add_column("Coin",      style="cyan",  width=14)
    done_tbl.add_column("Folds",     justify="center", width=7)
    done_tbl.add_column("Best Acc",  justify="right",  width=10)
    done_tbl.add_column("Avg Acc",   justify="right",  width=10)
    done_tbl.add_column("Grade",     width=8)

    for c in done_coins:
        folds  = c.get("folds", [])
        best   = c.get("best_acc", 0)
        avg    = sum(f["test_acc"] for f in folds)/len(folds) if folds else 0
        grade  = "A ★★★" if best>=0.60 else "B ★★" if best>=0.55 else "C ★" if best>=0.52 else "D"
        gcol   = "green" if best>=0.55 else "yellow" if best>=0.52 else "dim"
        done_tbl.add_row(
            c["symbol"], str(len(folds)),
            f"[{gcol}]{best:.4f}[/{gcol}]",
            f"{avg:.4f}", f"[{gcol}]{grade}[/{gcol}]"
        )
    if not done_coins:
        done_tbl.add_row("[dim]—[/dim]","[dim]none yet[/dim]","","","")

    # ── In-progress other coins ───────────────────────────────────────
    prog_tbl = Table(
        title="[bold]All Coins In Progress[/bold]",
        box=box.SIMPLE_HEAD, show_lines=False, expand=True
    )
    prog_tbl.add_column("Coin",  style="cyan", width=14)
    prog_tbl.add_column("Folds Done", justify="center", width=11)
    prog_tbl.add_column("Best Acc", justify="right", width=10)
    prog_tbl.add_column("Last Test", justify="right", width=10)

    for c in in_prog:
        folds = c.get("folds",[])
        last  = folds[-1]["test_acc"] if folds else 0
        best  = c.get("best_acc",0)
        prog_tbl.add_row(
            c["symbol"], f"{len(folds)}/10",
            f"{best:.4f}", f"{last:.4f}"
        )

    return header, progress, gpu_panel, fold_tbl, done_tbl, prog_tbl


def main():
    if not HAS_RICH:
        while True:
            try: plain_render(); time.sleep(REFRESH)
            except KeyboardInterrupt: print("\nStopped."); break
        return

    console = Console()
    with Live(console=console, refresh_per_second=0.2, screen=True) as live:
        while True:
            try:
                header, progress, gpu_panel, fold_tbl, done_tbl, prog_tbl = build_rich_layout()

                layout = Layout()
                layout.split_column(
                    Layout(header,                          name="hdr",  size=3),
                    Layout(Columns([progress, gpu_panel]),  name="top",  size=8),
                    Layout(fold_tbl,                        name="fold", ratio=3),
                    Layout(Columns([done_tbl, prog_tbl]),   name="bot",  ratio=2),
                )
                live.update(layout)
                time.sleep(REFRESH)

            except KeyboardInterrupt:
                console.print("\n[yellow]Monitor stopped.[/yellow]")
                break
            except Exception as e:
                time.sleep(REFRESH)


if __name__ == "__main__":
    main()
