#!/usr/bin/env python3
"""
RIGOROUS (UNCONDITIONAL) TAIL BOUNDS
====================================

Z-tail  : ∫_T^∞ |h(t)| · (c0 log t + c1) dt        using dN/dt <= c0 log t + c1  (t >= T0)
P-tail  : (psi bound) <= (Cψ/π) ∫_{log N}^∞ |ĥ(y)| e^{y/2} dy
A-tail  : (1/2π) ∫_{|t|>tmax} |h(t)| (log|t| + C_A) dt

These are deliberately unconditional and parameterised by BoundsConfig.
"""

from __future__ import annotations

import numpy as np
from math import log, pi
from dataclasses import dataclass
from typing import Callable, Dict, Any, Tuple

from scipy import integrate

from .bounds_config import BoundsConfig, DEFAULT_BOUNDS, warn_if_placeholders
from fourier_conventions import archimedean_integrand, sieve_primes


def z_tail_bound(h: Callable[[float], float], T: float, cfg: BoundsConfig = DEFAULT_BOUNDS) -> float:
    """
    Upper bound for Z-tail using dN/dt <= c0 log t + c1 for t >= T0.
    Bound: ∫_T^∞ |h(t)| (c0 log t + c1) dt.
    """
    T = max(T, cfg.T0)

    def integrand(t: float) -> float:
        dens = cfg.dN_c0 * log(max(t, cfg.T0)) + cfg.dN_c1
        return abs(h(t)) * dens

    val, _ = integrate.quad(integrand, T, np.inf, limit=400, epsrel=1e-8)
    return float(val)


def p_tail_bound(hhat: Callable[[float], float], N: int, cfg: BoundsConfig = DEFAULT_BOUNDS) -> float:
    """
    Upper bound for P-tail via psi(x) <= Cψ x (x >= ψ_x0).
    Tail <= (Cψ/π) * ∫_{log max(N, ψ_x0)}^∞ |ĥ(y)| e^{y/2} dy
    (there is an extra factor 2/(2π) = 1/π from the explicit formula's prime term).
    """
    y0 = log(max(int(N), int(cfg.psi_x0)))

    def integrand(y: float) -> float:
        return abs(hhat(y)) * np.exp(0.5 * y)

    val, _ = integrate.quad(integrand, y0, np.inf, limit=400, epsrel=1e-8)
    return float((cfg.psi_C / pi) * val)


def a_tail_bound(h: Callable[[float], float], tmax: float, cfg: BoundsConfig = DEFAULT_BOUNDS) -> float:
    """
    Upper bound for A-tail using |Re ψ(1/4 + it/2) - log π| <= log|t| + C_A for |t| >= A_tail_x0.
    Tail <= (1/2π) * 2 * ∫_{tmax}^∞ |h(t)| (log t + C_A) dt  (even h)
    """
    t0 = max(tmax, cfg.A_tail_x0)

    def integrand(t: float) -> float:
        return abs(h(t)) * (log(max(t, cfg.A_tail_x0)) + cfg.A_tail_C) / (2 * pi)

    val, _ = integrate.quad(integrand, t0, np.inf, limit=400, epsrel=1e-8)
    return float(2.0 * val)


def compute_prime_partial(hhat: Callable[[float], float], N: int) -> float:
    """
    Truncated prime term:
      P_{<=N} = (1/2π) * Σ_{p^k <= N} 2 (log p) / p^{k/2} * ĥ(k log p)
    """
    P = 0.0
    primes = sieve_primes(max(1000, int(N)))
    for p in primes:
        if p > N:
            break
        logp = log(p)
        pk = p
        k = 1
        while pk <= N:
            P += 2.0 * (logp / (pk ** 0.5)) * hhat(k * logp)
            pk *= p
            k += 1
    return float(P / (2 * pi))


def compute_archimedean_truncated(h: Callable[[float], float], tmax: float) -> Tuple[float, float]:
    """Truncated A on [-tmax, tmax] using the exact integrand; returns (A_partial, quad_err)."""

    def integrand(t: float) -> float:
        return archimedean_integrand(t, h)

    A_partial, err = integrate.quad(integrand, -tmax, tmax, limit=400, epsrel=1e-8)
    return float(A_partial), float(err)


def compute_z_truncated(h: Callable[[float], float], zeros: np.ndarray, include_negative_zeros: bool = True) -> float:
    """Z_{<=T} using supplied zeros (positive ordinates)."""
    from fourier_conventions import ZeroSumConvention

    return float(ZeroSumConvention.z_term_standard(h, list(zeros), include_negative_zeros))


def compute_Q_truncated_with_bounds(
    h: Callable[[float], float],
    hhat: Callable[[float], float],
    zeros: np.ndarray,
    N_max: int,
    t_max: float,
    cfg: BoundsConfig = DEFAULT_BOUNDS,
    include_negative_zeros: bool = True,
) -> Dict[str, Any]:
    """
    Returns dict with:
      Q_T, Z_T, A_T, P_T, tail_Z, tail_A, tail_P, tail_total, Q_lower
    """
    warn_if_placeholders(cfg)

    Z_T = compute_z_truncated(h, zeros, include_negative_zeros)
    A_T, A_err = compute_archimedean_truncated(h, t_max)
    P_T = compute_prime_partial(hhat, int(N_max))

    gamma_max = float(zeros[-1]) if len(zeros) else cfg.T0
    tail_Z = z_tail_bound(h, gamma_max, cfg)
    tail_A = a_tail_bound(h, t_max, cfg)
    tail_P = p_tail_bound(hhat, int(N_max), cfg)
    tail_total = tail_Z + tail_A + tail_P

    Q_T = Z_T - A_T - P_T
    Q_lower = Q_T - tail_total

    return {
        "Z_T": Z_T,
        "A_T": A_T,
        "P_T": P_T,
        "tail_Z": tail_Z,
        "tail_A": tail_A,
        "tail_P": tail_P,
        "tail_total": tail_total,
        "Q_T": Q_T,
        "Q_lower": Q_lower,
        "A_quad_err": A_err,
        "gamma_max": gamma_max,
        "N_max": int(N_max),
        "t_max": float(t_max),
    }

