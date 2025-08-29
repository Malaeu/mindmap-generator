#!/usr/bin/env python3
"""
WEIL TRUNCATION CERTIFICATE (UNCONDITIONAL)
===========================================
Compute Q_T(h) with unconditional tail bounds and certify Q_lower > 0.

Usage examples:
  python3 -m experiments.weil_truncated --zeros=100000 --family gaussian --grid "0.6:1.4:0.02"
  python3 -m experiments.weil_truncated --zeros=100000 --family autocorr --param 3.0

Outputs CSV and a summary table. Optionally draws a quick plot.

NOTE: Constants live in bounds_config.py and MUST be set to literature-backed values
for a final unconditional claim.
"""

from __future__ import annotations

import argparse
import csv
from typing import List

import numpy as np
from rich.console import Console
from rich.table import Table
from rich import box

import matplotlib.pyplot as plt

from experiments.rigorous_bounds import compute_Q_truncated_with_bounds
from experiments.bounds_config import DEFAULT_BOUNDS
from experiments.certified_validation import load_zeros as load_zeros_default


console = Console()


def parse_grid(spec: str) -> List[float]:
    a, b, c = [float(x) for x in spec.split(":")]
    n = max(1, int(round((b - a) / c)) + 1)
    return [a + i * c for i in range(n)]


def make_pair(family: str, param: float):
    import math
    import numpy as np

    fam = family.lower()
    if fam == "gaussian":
        sigma = float(param)
        def h(t: float) -> float:
            return math.exp(-(t * t) / (2.0 * sigma * sigma))
        def hhat(u: float) -> float:
            return math.sqrt(2.0 * math.pi) * sigma * math.exp(-(sigma * sigma) * (u * u) / 2.0)
        return h, hhat, f"Gaussian σ={sigma}"
    elif fam == "cauchy":
        beta = float(param)
        def h(t: float) -> float:
            return 1.0 / (1.0 + (t / beta) ** 2)
        def hhat(u: float) -> float:
            return math.pi * beta * math.exp(-beta * abs(u))
        return h, hhat, f"Cauchy β={beta}"
    elif fam == "bump":
        beta = float(param)
        def h(t: float) -> float:
            if t == 0.0:
                return beta * beta
            s = math.sin(beta * t)
            return (s / t) * (s / t)
        def hhat(u: float) -> float:
            val = max(0.0, 2.0 * beta - abs(u))
            return math.pi * val
        return h, hhat, f"Bump β={beta}"
    elif fam == "autocorr":
        sigma = float(param)
        def h(t: float) -> float:
            return math.sqrt(math.pi) * sigma * math.exp(-(t * t) / (4.0 * sigma * sigma))
        def hhat(u: float) -> float:
            return 2.0 * math.pi * (sigma * sigma) * math.exp(-(sigma * sigma) * (u * u))
        return h, hhat, f"Autocorr(Gauss σ={sigma})"
    else:
        raise ValueError(f"Unknown family: {family}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Truncated Weil certificate (unconditional tails)")
    ap.add_argument("--zeros", type=int, default=100000, help="How many zeros to load")
    ap.add_argument("--zeros-file", type=str, default=None, help="Optional zeros file (one γ per line)")
    ap.add_argument("--family", type=str, default="gaussian", choices=["gaussian", "cauchy", "bump", "autocorr"], help="Test family")
    ap.add_argument("--grid", type=str, default="0.6:1.4:0.02", help="Grid spec start:stop:step")
    ap.add_argument("--param", type=float, default=None, help="Single parameter value (overrides grid)")
    ap.add_argument("--N-max", type=int, default=100000, help="Prime cutoff N for P_term")
    ap.add_argument("--t-max", type=float, default=50.0, help="Time cutoff for A_term")
    ap.add_argument("--csv", type=str, default="truncation_certificate.csv")
    ap.add_argument("--plot", action="store_true", help="Quick plot of Q_lower vs parameter")
    args = ap.parse_args()

    zeros = load_zeros_default(args.zeros, args.zeros_file)
    zeros_arr = np.array(zeros, dtype=float)
    params = [args.param] if args.param is not None else parse_grid(args.grid)

    table = Table(box=box.SIMPLE)
    table.add_column("param", justify="center")
    table.add_column("Q_T", justify="right")
    table.add_column("Tail", justify="right")
    table.add_column("Q_lower", justify="right")
    table.add_column("Status", justify="center")

    rows = []
    positives = 0

    for p in params:
        h, hhat, label = make_pair(args.family, p)
        out = compute_Q_truncated_with_bounds(
            h, hhat, zeros_arr, N_max=args.N_max, t_max=args.t_max, cfg=DEFAULT_BOUNDS
        )
        Q_T = out["Q_T"]; tail = out["tail_total"]; Q_lower = out["Q_lower"]
        ok = Q_lower > 0
        positives += int(ok)
        status = "[green]✅[/green]" if ok else "[red]❌[/red]"
        q_color = "green" if ok else "red"
        table.add_row(f"{p:.3f}", f"{Q_T:+.5f}", f"{tail:.2e}", f"[{q_color}]{Q_lower:+.5f}[/{q_color}]", status)
        rows.append({"param": p, "Q_T": Q_T, "tail_total": tail, "Q_lower": Q_lower, "pass": ok})

    console.print(f"[bold cyan]TRUNCATION CERTIFICATE[/bold cyan]  family={args.family}, zeros={len(zeros)}, γ_max≈{zeros_arr[-1] if len(zeros_arr)>0 else 0:.1f}")
    console.print(table)
    rate = positives / len(params) if params else 0.0
    console.print(f"\n[bold]Pass:[/bold] {positives}/{len(params)}  ({rate:.1%})")

    with open(args.csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["param", "Q_T", "tail_total", "Q_lower", "pass"])
        w.writeheader()
        w.writerows(rows)
    console.print(f"[dim]Saved CSV to {args.csv}[/dim]
")

    if args.plot and rows:
        xs = [r["param"] for r in rows]
        ys = [r["Q_lower"] for r in rows]
        plt.figure(figsize=(8, 5))
        plt.axhline(0.0, color="k", linestyle="--", linewidth=2, alpha=0.5)
        plt.plot(xs, ys, "o-", linewidth=2)
        plt.xlabel("parameter")
        plt.ylabel("Q_lower")
        plt.title(f"Q_lower vs parameter [{args.family}]")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("truncation_q_lower.png", dpi=150, bbox_inches="tight")
        console.print("[dim]Saved plot to truncation_q_lower.png[/dim]")


if __name__ == "__main__":
    main()

