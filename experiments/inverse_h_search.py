#!/usr/bin/env python3
"""
INVERSE H SEARCH (adversarial)
==============================
Given truncation parameters (zeros, N_max, t_max), search over function families
to minimise Q_lower and report near-failure cases.
"""

from __future__ import annotations

import argparse
from typing import List

import numpy as np
from rich.console import Console
from rich.table import Table
from rich import box
import matplotlib.pyplot as plt

from experiments.rigorous_bounds import compute_Q_truncated_with_bounds
from experiments.bounds_config import DEFAULT_BOUNDS
from experiments.weil_truncated import make_pair
from experiments.certified_validation import load_zeros as load_zeros_default


console = Console()


def parse_grid(spec: str) -> List[float]:
    a, b, c = [float(x) for x in spec.split(":")]
    n = max(1, int(round((b - a) / c)) + 1)
    return [a + i * c for i in range(n)]


def main() -> None:
    ap = argparse.ArgumentParser(description="Adversarial search for minimal Q_lower")
    ap.add_argument("--zeros", type=int, default=100000)
    ap.add_argument("--zeros-file", type=str, default=None)
    ap.add_argument("--N-max", type=int, default=100000)
    ap.add_argument("--t-max", type=float, default=50.0)
    ap.add_argument("--family", type=str, default="gaussian", choices=["gaussian", "autocorr"]) 
    ap.add_argument("--sigma-grid", type=str, default="0.6:1.4:0.01")
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--heatmap", action="store_true")
    args = ap.parse_args()

    zeros = np.array(load_zeros_default(args.zeros, args.zeros_file), dtype=float)
    sigmas = parse_grid(args.sigma_grid)

    candidates = []
    for s in sigmas:
        h, hhat, label = make_pair(args.family, s)
        out = compute_Q_truncated_with_bounds(h, hhat, zeros, args.N_max, args.t_max, DEFAULT_BOUNDS)
        candidates.append((s, out["Q_lower"], label))

    candidates.sort(key=lambda x: x[1])
    table = Table(box=box.SIMPLE)
    table.add_column("rank", justify="right")
    table.add_column("label", justify="left")
    table.add_column("Q_lower", justify="right")
    for i, (s, ql, label) in enumerate(candidates[: args.topk], 1):
        color = "green" if ql > 0 else "red"
        table.add_row(str(i), label, f"[{color}]{ql:+.6f}[/{color}]")
    console.print(table)

    if args.heatmap:
        xs = [s for (s, _, _) in candidates]
        ys = [ql for (_, ql, _) in candidates]
        plt.figure(figsize=(8, 4))
        plt.plot(xs, ys, "-", linewidth=2)
        plt.axhline(0.0, color="k", linestyle="--", linewidth=1.5)
        plt.xlabel("σ")
        plt.ylabel("Q_lower")
        plt.title(f"Adversarial search [{args.family}], zeros={len(zeros)}")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("inverse_search_q_lower.png", dpi=150, bbox_inches="tight")
        console.print("[dim]Saved plot to inverse_search_q_lower.png[/dim]")


if __name__ == "__main__":
    main()

