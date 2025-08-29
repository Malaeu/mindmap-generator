"""
Test function families h and Fourier transforms ĥ for certified validation.

Convention: ĥ(ξ) = ∫ h(t) e^{-i ξ t} dt.
All functions accept mp.iv.mpf (interval) inputs and return interval outputs.
"""

from __future__ import annotations

from collections.abc import Callable

import mpmath as mp

mp.mp.dps = 60

def iv_const(x: float) -> mp.iv.mpf:
    return mp.iv.mpf([float(x), float(x)])

def iv_abs(x: mp.iv.mpf) -> mp.iv.mpf:
    a = float(x.a)
    b = float(x.b)
    if a <= 0.0 <= b:
        return mp.iv.mpf([0.0, max(abs(a), abs(b))])
    lo = min(abs(a), abs(b))
    hi = max(abs(a), abs(b))
    return mp.iv.mpf([lo, hi])


def gaussian_family(sigma: float) -> tuple[Callable[[mp.iv.mpf], mp.iv.mpf], Callable[[mp.iv.mpf], mp.iv.mpf]]:
    s = float(sigma)
    s_iv = iv_const(s)
    sqrt2pi = mp.iv.sqrt(iv_const(2.0 * mp.pi))
    def h(t: mp.iv.mpf) -> mp.iv.mpf:
        return mp.iv.exp(- (t * t) / (2.0 * s_iv * s_iv))
    def hhat(xi: mp.iv.mpf) -> mp.iv.mpf:
        return sqrt2pi * s_iv * mp.iv.exp(- (s_iv * s_iv) * (xi * xi) / 2.0)
    return h, hhat


def cauchy_family(sigma: float) -> tuple[Callable[[mp.iv.mpf], mp.iv.mpf], Callable[[mp.iv.mpf], mp.iv.mpf]]:
    """
    h(t) = 1 / (1 + (t/σ)^2). Fourier transform (our convention): ĥ(ξ) = π σ e^{-σ |ξ|}.
    """
    s = float(sigma)
    s_iv = iv_const(s)
    pi_iv = iv_const(float(mp.pi))
    def h(t: mp.iv.mpf) -> mp.iv.mpf:
        return 1.0 / (1.0 + (t / s_iv) ** 2)
    def hhat(xi: mp.iv.mpf) -> mp.iv.mpf:
        # |xi| as interval: conservative via sup(|bounds|)
        xi_abs = iv_abs(xi)
        return pi_iv * s_iv * mp.iv.exp(- (s_iv * xi_abs))
    return h, hhat


def bump_family(beta: float) -> tuple[Callable[[mp.iv.mpf], mp.iv.mpf], Callable[[mp.iv.mpf], mp.iv.mpf]]:
    """
    Fejér-type bump: h(t) = (sin(β t)/t)^2,   ĥ(ξ) = π (2β - |ξ|)_+.
    Both nonnegative and even; ĥ≥0 by construction.
    """
    b = float(beta)
    pi_iv = iv_const(float(mp.pi))

    def h(t: mp.iv.mpf) -> mp.iv.mpf:
        # If interval crosses 0, return [0, β^2] safely
        if float(t.a) <= 0.0 <= float(t.b):
            return mp.iv.mpf([0.0, b * b])
        s = mp.iv.sin(b * t)
        q = s / t
        return q * q

    def hhat(xi: mp.iv.mpf) -> mp.iv.mpf:
        a = abs(float(xi.a)); c = abs(float(xi.b))
        lo_abs = min(a, c)
        hi_abs = max(a, c)
        # Range of (2b - u)_+ over u∈[lo_abs, hi_abs]
        vals = [max(0.0, 2*b - lo_abs), max(0.0, 2*b - hi_abs)]
        if lo_abs <= 2*b <= hi_abs:
            vals.append(0.0)
        low = min(vals); high = max(vals)
        return pi_iv * mp.iv.mpf([low, high])

    return h, hhat


def autocorr_gaussian_family(sigma_g: float) -> tuple[Callable[[mp.iv.mpf], mp.iv.mpf], Callable[[mp.iv.mpf], mp.iv.mpf]]:
    """
    Autocorrelation of Gaussian g_σ: h = g * g~,  ĥ = |ĥ_g|^2.
    With our convention: ĥ_g(ξ)=√(2π)σ e^{-σ^2 ξ^2/2} ⇒ ĥ(ξ)=2π σ^2 e^{-σ^2 ξ^2}.
    Inverse gives h(t)=√π σ e^{-t^2/(4 σ^2)}.
    """
    s = float(sigma_g)
    s_iv = iv_const(s)
    sqrt_pi = mp.iv.sqrt(iv_const(float(mp.pi)))
    two_pi = iv_const(2.0 * float(mp.pi))

    def h(t: mp.iv.mpf) -> mp.iv.mpf:
        return sqrt_pi * s_iv * mp.iv.exp(- (t * t) / (4.0 * s_iv * s_iv))

    def hhat(xi: mp.iv.mpf) -> mp.iv.mpf:
        return two_pi * (s_iv * s_iv) * mp.iv.exp(- (s_iv * s_iv) * (xi * xi))

    return h, hhat


def hhat_abs_from(hhat: Callable[[mp.iv.mpf], mp.iv.mpf]) -> Callable[[mp.iv.mpf], mp.iv.mpf]:
    def f(u: mp.iv.mpf) -> mp.iv.mpf:
        val = hhat(u)
        return iv_abs(val)
    return f


def make_family(name: str, **params) -> tuple[Callable, Callable]:
    name = name.lower()
    if name == "gaussian":
        sigma = float(params.get("sigma", 1.0))
        return gaussian_family(sigma)
    if name == "cauchy":
        sigma = float(params.get("sigma", 1.0))
        return cauchy_family(sigma)
    if name == "bump":
        beta = float(params.get("sigma", 1.0))
        return bump_family(beta)
    if name == "autocorr":
        sigma = float(params.get("sigma", 1.0))
        return autocorr_gaussian_family(sigma)
    raise ValueError(f"Unsupported family: {name}")
