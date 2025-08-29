#!/usr/bin/env python3
from __future__ import annotations

import math
from collections.abc import Callable
import mpmath as mp


def rho_up_factory(c0: float = 0.28) -> Callable[[float], float]:
    def rho_up(t: float) -> float:
        if t <= 0:
            return 0.0
        return max(0.0, (math.log(t / (2.0 * math.pi)) / (2.0 * math.pi)) + c0)
    return rho_up


def compute_L_sigma_box(
    family: str,
    sigma_lo: float,
    sigma_hi: float,
    zeros: list[float],
    Gamma: float,
    p_cut: int = 10000,
    A_envelope_const: float = 4.0,
    c0: float = 0.28,
) -> float:
    """
    Returns L such that |Q(σ)-Q(σ')| ≤ L |σ-σ'| for σ,σ' in [sigma_lo, sigma_hi].
    Implemented for gaussian and autocorr (gaussian-based) families.
    """
    sig = 0.5 * (sigma_lo + sigma_hi)
    rho_up = rho_up_factory(c0)

    if family not in ("gaussian", "autocorr"):
        raise ValueError("Lipschitz bound currently implemented for gaussian/autocorr families only")

    # Gaussian core definitions
    def h_gauss(t, s=sig):
        return mp.e ** (-(t * t) / (2 * s * s))

    # Z-part derivative bound
    Z_disc = sum((g * g) * h_gauss(g) for g in zeros if g <= Gamma)
    Z_tail = mp.quad(lambda t: (t * t) * h_gauss(t) * rho_up(float(t)), [Gamma, mp.inf])
    L_Z = (2.0 / (sig ** 3)) * (Z_disc + Z_tail)

    # A-part derivative bound using |M(t)| ≤ log(1+|t|)+C
    A_env = lambda t: (t * t) * h_gauss(t) * (mp.log(1 + abs(t)) + A_envelope_const) / (2 * mp.pi * sig ** 3)
    L_A = 2 * mp.quad(A_env, [0, mp.inf])

    # P-part derivative bound
    sqrt2pi = math.sqrt(2.0 * math.pi)
    def d_hat_gauss(u, s=sig):
        return sqrt2pi * mp.e ** (-(s * s) * (u * u) / 2.0) * (1 - s * s * u * u)

    disc = 0.0
    # simple over-bound via all n≤p_cut (can be optimized to primes only)
    for n in range(2, p_cut + 1):
        disc += (mp.log(n) / mp.sqrt(n)) * abs(d_hat_gauss(mp.log(n)))
    L_P_disc = (2.0 / mp.pi) * disc
    L_P_tail = (1.2 / mp.pi) * mp.quad(lambda u: mp.e ** (u / 2.0) * abs(d_hat_gauss(u)), [mp.log(p_cut), mp.inf])
    L_P = L_P_disc + L_P_tail

    return float(L_Z + L_A + L_P)

