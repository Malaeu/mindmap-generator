#!/usr/bin/env python3
"""
Certified validation CLI (Step 1): interval bounds for Q = Z - A - P.

Usage:
  python -m experiments.certified_validation \
    --family {gaussian,cauchy} --zeros 1000 --grid phase3 --json out.json

Outputs human-readable lines per node and writes JSON certificates with
component interval decompositions and tail bounds.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import asdict, dataclass
from typing import Any
import concurrent.futures as futures
import multiprocessing as mp

from experiments.families import hhat_abs_from, make_family
from experiments.interval_utils import (
    IntervalResult,
    archimedean_interval,
    archimedean_interval_tight,
    prime_sum_interval,
    prime_tail_interval,
    z_sum_interval,
    z_tail_interval,
)
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

# Defaults
DEFAULT_ZEROS: list[float] = [
    14.134725, 21.022040, 25.010858, 30.424876, 32.935062,
    37.586178, 40.918719, 43.327073, 48.005151, 49.773832,
    52.970321, 56.446248, 59.347044, 60.831779, 65.112544,
    67.079811, 69.546402, 72.067158, 75.704691, 77.144840,
    79.337375, 82.910381, 84.735493, 87.425275, 88.809111,
    92.491899, 94.651344, 95.870634, 98.831194, 101.317851
]


def load_zeros(count: int, path: str | None = None) -> list[float]:
    if path and os.path.exists(path):
        zs: list[float] = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    zs.append(float(line))
                    if len(zs) >= count:
                        break
                except ValueError:
                    continue
        if zs:
            return zs[:count]
    # fallback
    return DEFAULT_ZEROS[: min(count, len(DEFAULT_ZEROS))]


def grid_sigmas(spec: str) -> list[float]:
    spec = spec.lower()
    if spec in ("phase3", "default"):
        # Dense near transition ~0.97 and broad coverage
        A = [round(x, 3) for x in [0.5 + 0.05 * i for i in range(0, 11)]]  # 0.5..1.0 step 0.05
        B = [round(x, 2) for x in [1.0 + 0.1 * i for i in range(0, 21)]]   # 1.0..3.0 step 0.1
        C = [round(x, 1) for x in [3.0 + 0.2 * i for i in range(0, 36)]]   # 3.0..10.0 step 0.2
        return sorted(set(A + B + C))
    # custom comma list
    if "," in spec:
        return [float(x) for x in spec.split(",") if x.strip()]
    # single value
    try:
        return [float(spec)]
    except ValueError:
        raise SystemExit(f"Unsupported grid spec: {spec}") from None


@dataclass
class ComponentCertificate:
    interval: tuple[float, float]
    meta: dict[str, Any]


@dataclass
class NodeCertificate:
    family: str
    sigma: float
    zeros_used: int
    Q_lower: float
    Q_upper: float
    Z: tuple[float, float]
    A: tuple[float, float]
    P: tuple[float, float]
    tails: dict[str, float]


def certify_node(
    family: str,
    sigma: float,
    zeros: list[float],
    z_tail_c0: float = 0.25,
    p_N: int = 5000,
    a_N_main: int = 800,
    p_M: int = 800,
    z_M: int = 800,
    progress: Progress | None = None,
) -> NodeCertificate:
    # Build family
    h, hhat = make_family(family, sigma=sigma)
    hhat_abs = hhat_abs_from(hhat)

    # Z main and tail
    subtask = None
    if progress is not None:
        subtask = progress.add_task(f"σ={sigma:.3g} [1/5] Z-main", total=5)
    Z_main = z_sum_interval(h, zeros)
    if subtask is not None:
        progress.update(subtask, advance=1, description=f"σ={sigma:.3g} [2/5] Z-tail")
    Z_tail = z_tail_interval(h, zeros[-1] if zeros else 0.0, c0=z_tail_c0, M=int(z_M))
    Z = IntervalResult(Z_main.lower, Z_main.upper + Z_tail.upper)

    # A-term: choose t_max relative to sigma
    t_max = max(10.0 * sigma, 40.0)
    if subtask is not None:
        progress.update(subtask, advance=1, description=f"σ={sigma:.3g} [3/5] A-main (0/{int(a_N_main)})")
    # progress callback for A-main chunks (updates description with i/N)
    def a_progress(done: int, total: int):
        if subtask is not None:
            progress.update(subtask, description=f"σ={sigma:.3g} [3/5] A-main ({done}/{total})")
    # Use tight Archimedean interval by default
    A_total, A_meta, (A_pos, A_neg) = archimedean_interval_tight(
        h, t_max=t_max, N_main=int(a_N_main), progress_cb=a_progress
    )

    # P main and tail
    if subtask is not None:
        progress.update(subtask, advance=1, description=f"σ={sigma:.3g} [4/5] P-main")
    P_main, P_terms = prime_sum_interval(hhat, N=int(p_N))
    # Tail upper via integral in u-space
    u_max = max(math.log(max(2, int(p_N))) + 20.0, 25.0)
    if subtask is not None:
        progress.update(subtask, description=f"σ={sigma:.3g} [5/5] P-tail")
    P_tail = prime_tail_interval(hhat_abs, N=int(p_N), u_max=u_max, M=int(p_M), A_psi=1.2)
    P = IntervalResult(P_main.lower, P_main.upper + P_tail.upper)

    # Q = Z - A - P
    Q_lo = Z.lower - A_total.upper - P.upper
    Q_hi = Z.upper - A_total.lower - P.lower
    if subtask is not None:
        progress.update(subtask, advance=1, description=f"σ={sigma:.3g} done")
        progress.remove_task(subtask)

    return NodeCertificate(
        family=family,
        sigma=float(sigma),
        zeros_used=len(zeros),
        Q_lower=float(Q_lo),
        Q_upper=float(Q_hi),
        Z=Z.to_tuple(),
        A=A_total.to_tuple(),
        P=P.to_tuple(),
        tails={
            "Z_tail_upper": float(Z_tail.upper),
            "A_tail_upper": float(A_meta.get("tail_upper", 0.0)),
            "P_tail_upper": float(P_tail.upper),
        },
    )


# Top-level worker for parallel execution (must be picklable)
def _certify_worker(payload: dict) -> dict:
    family = payload["family"]
    sigma = payload["sigma"]
    zeros = payload["zeros"]
    p_N = payload["p_N"]
    a_N_main = payload["a_N_main"]
    p_M = payload["p_M"]
    z_M = payload["z_M"]
    cert = certify_node(
        family, sigma, zeros, p_N=p_N,
        a_N_main=a_N_main, p_M=p_M, z_M=z_M,
        progress=None,
    )
    return asdict(cert)


def run_cli() -> None:
    ap = argparse.ArgumentParser(description="Certified validation (Step 1)")
    ap.add_argument("--family", required=True, choices=["gaussian", "cauchy", "bump", "autocorr"], help="Test family")
    ap.add_argument("--zeros", type=int, default=1000, help="How many zeros to use")
    ap.add_argument("--zeros-file", type=str, default=None, help="Optional path to zeros list (one γ per line)")
    ap.add_argument("--grid", type=str, default="phase3", help="Grid spec (phase3 or comma list)")
    ap.add_argument("--json", type=str, default=None, help="Output JSON file for certificates")
    ap.add_argument("--p-cut", type=int, default=5000, help="Prime cutoff N for main P-sum")
    ap.add_argument("--a-main", type=int, default=800, help="A-term main subintervals (N_main)")
    ap.add_argument("--p-tail-m", type=int, default=800, help="P-tail subintervals (M)")
    ap.add_argument("--z-tail-m", type=int, default=800, help="Z-tail subintervals (M)")
    ap.add_argument("--workers", type=int, default=1, help="Parallel workers (processes); 1 = sequential with detailed sub-steps")
    args = ap.parse_args()

    # All families supported: gaussian, cauchy, bump, autocorr

    sigmas = grid_sigmas(args.grid)
    zeros = load_zeros(args.zeros, args.zeros_file)

    out: dict[str, Any] = {
        "family": args.family,
        "zeros": len(zeros),
        "grid": sigmas,
        "nodes": [],
    }

    # Parallel or sequential execution
    if args.workers and args.workers > 1:
        max_workers = min(args.workers, mp.cpu_count() or args.workers)
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=None),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            transient=False,
        ) as progress:
            main = progress.add_task(f"Family={args.family} | Nodes={len(sigmas)} | Workers={max_workers}", total=len(sigmas))
            results: list[dict] = []
            with futures.ProcessPoolExecutor(max_workers=max_workers) as ex:
                fut_map = {}
                for s in sigmas:
                    payload = {
                        "family": args.family,
                        "sigma": s,
                        "zeros": zeros,
                        "p_N": args.p_cut,
                        "a_N_main": args.a_main,
                        "p_M": args.p_tail_m,
                        "z_M": args.z_tail_m,
                    }
                    fut_map[ex.submit(_certify_worker, payload)] = s
                for i, fut in enumerate(futures.as_completed(fut_map), 1):
                    cert_d = fut.result()
                    s = fut_map[fut]
                    sign = "+" if cert_d["Q_lower"] >= 0 else ("?" if cert_d["Q_upper"] >= 0 else "-")
                    print(
                        f"sigma: {s:.6g}  Q in [{cert_d['Q_lower']:+.6e}, {cert_d['Q_upper']:+.6e}]  "
                        f"(Z={tuple(cert_d['Z'])}, A={tuple(cert_d['A'])}, P={tuple(cert_d['P'])}, tails <= {cert_d['tails']})  [{sign}]"
                    )
                    results.append(cert_d)
                    progress.advance(main)
            # keep JSON nodes in sigma order
            out["nodes"].extend(sorted(results, key=lambda c: c["sigma"]))
    else:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=None),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            transient=False,
        ) as progress:
            main = progress.add_task(f"Family={args.family} | Nodes={len(sigmas)}", total=len(sigmas))
            for idx, s in enumerate(sigmas, 1):
                progress.update(main, description=f"{idx}/{len(sigmas)} Processing σ={s:.3g}")
                cert = certify_node(
                    args.family, s, zeros, p_N=args.p_cut,
                    a_N_main=args.a_main, p_M=args.p_tail_m, z_M=args.z_tail_m,
                    progress=progress,
                )
                sign = "+" if cert.Q_lower >= 0 else ("?" if cert.Q_upper >= 0 else "-")
                print(
                    f"sigma: {s:.6g}  Q in [{cert.Q_lower:+.6e}, {cert.Q_upper:+.6e}]  "
                    f"(Z={cert.Z}, A={cert.A}, P={cert.P}, tails <= {cert.tails})  [{sign}]"
                )
                out["nodes"].append(asdict(cert))
                progress.advance(main)

    if args.json:
        os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print(f"\nSaved certificates to {args.json}")


if __name__ == "__main__":
    run_cli()
