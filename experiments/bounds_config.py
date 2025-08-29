#!/usr/bin/env python3
"""
Bounds configuration (UNCONDITIONAL placeholders).

This module stores explicit constants controlling unconditional tail bounds.
Replace placeholders by explicit literature-backed constants (e.g. Rosser–Schoenfeld,
Dusart, Trudgian, etc.) before claiming a final unconditional certificate.

All constants are used as upper bounds; choosing them larger is safe
(but gives looser tails).

IMPORTANT: Do NOT hardcode RH or average density; these are unconditional knobs.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BoundsConfig:
    # Z-tail via zero density bound: dN/dt <= dN_c0 * log t + dN_c1 for t >= T0
    T0: float = 100.0
    dN_c0: float = 0.17     # ≈ 1/(2π)=0.159...  (SAFE placeholder; adjust upward if uncertain)
    dN_c1: float = 0.00     # extra slack

    # P-tail via Chebyshev psi: psi(x) <= psi_C * x for x >= psi_x0
    psi_C: float = 1.10     # SAFE placeholder (use explicit constant from literature)
    psi_x0: float = 100.0

    # A-tail via digamma: |Re ψ(1/4 + i t/2) - log π| <= log|t| + A_tail_C for |t| >= A_tail_x0
    A_tail_C: float = 2.0   # generous slack
    A_tail_x0: float = 10.0


DEFAULT_BOUNDS = BoundsConfig()


def warn_if_placeholders(cfg: BoundsConfig = DEFAULT_BOUNDS) -> None:
    import warnings
    msg = (
        "[Phase IV] Using placeholder unconditional constants in bounds_config.py.\n"
        "Replace them with explicit, literature-backed values before claiming a final certificate."
    )
    warnings.warn(msg, RuntimeWarning)

