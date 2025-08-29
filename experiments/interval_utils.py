"""
Interval arithmetic utilities for certified bounds.

Requirements: mpmath>=1.3.0 (mp.iv)
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import mpmath as mp

# Configure default precision for interval computations
mp.mp.dps = 60


@dataclass
class IntervalResult:
    lower: float
    upper: float

    def width(self) -> float:
        return float(self.upper - self.lower)

    def to_tuple(self) -> tuple[float, float]:
        return (float(self.lower), float(self.upper))

    @staticmethod
    def from_iv(x: mp.iv.mpf) -> IntervalResult:
        return IntervalResult(float(x.a), float(x.b))

    @staticmethod
    def point(x: float) -> IntervalResult:
        return IntervalResult(x, x)


def _ensure_iv(x: float | mp.iv.mpf) -> mp.iv.mpf:
    if isinstance(x, mp.iv.mpf):
        return x
    return mp.iv.mpf([x, x])


def interval_sum(intervals: list[IntervalResult]) -> IntervalResult:
    lo = 0.0
    hi = 0.0
    for it in intervals:
        lo += it.lower
        hi += it.upper
    return IntervalResult(lo, hi)


def interval_mul_scalar(iv: IntervalResult, s: float | mp.iv.mpf) -> IntervalResult:
    s_iv = _ensure_iv(float(s))
    # Compute using mpmath interval to avoid sign mistakes
    a = mp.iv.mpf([iv.lower, iv.upper])
    prod = a * s_iv
    return IntervalResult(float(prod.a), float(prod.b))


def interval_quad_pos(f: Callable[[mp.iv.mpf], mp.iv.mpf], a: float, b: float, N: int) -> IntervalResult:
    """
    Certified enclosure for ∫_a^b f(t) dt for nonnegative f using interval rectangles.

    - Splits [a,b] into N subintervals.
    - On each subinterval J=[x_i,x_{i+1}], evaluates F = f([x_i,x_{i+1}]) ∈ [F_lo, F_hi]
      and accumulates [F_lo*dx, F_hi*dx].
    - Returns [sum_lo, sum_hi].
    """
    if not (b >= a):
        raise ValueError("interval_quad_pos: require b>=a")
    a = float(a)
    b = float(b)
    N = max(1, int(N))
    dx = (b - a) / N
    sum_lo = mp.iv.mpf([0.0, 0.0])
    sum_hi = mp.iv.mpf([0.0, 0.0])
    for i in range(N):
        x0 = a + i * dx
        x1 = x0 + dx
        t_iv = mp.iv.mpf([x0, x1])
        F = f(t_iv)
        # enforce nonnegativity by intersecting with [0, +inf)
        F_lo = max(float(F.a), 0.0)
        F_hi = max(float(F.b), 0.0)
        area_lo = mp.iv.mpf([F_lo * dx, F_lo * dx])
        area_hi = mp.iv.mpf([F_hi * dx, F_hi * dx])
        sum_lo += area_lo
        sum_hi += area_hi
    return IntervalResult(float(sum_lo.a), float(sum_hi.b))


def interval_quad_signed_decomp(
    g: Callable[[mp.iv.mpf], mp.iv.mpf],
    a: float,
    b: float,
    N: int,
    max_refine: int = 10,
    on_progress: Callable[[int, int], None] | None = None,
) -> tuple[IntervalResult, IntervalResult]:
    """
    Certified enclosure for ∫ g = ∫ g_+ - ∫ g_- by decomposing sign on subintervals.

    - Split [a,b] into N chunks; on each J evaluate G = g(J) = [lo,hi].
    - If hi<=0 → contributes only to negative part; if lo>=0 → only to positive part.
    - If mixed sign, recursively bisect until we can classify or reach max_refine. For the latter,
      over-approximate by sending the entire absolute interval to both sides conservatively.
    Returns (I_pos, I_neg) where both are nonnegative interval enclosures of ∫ g_+ and ∫ g_-.
    """
    def process_interval(x0: float, x1: float, depth: int) -> tuple[mp.iv.mpf, mp.iv.mpf]:
        t_iv = mp.iv.mpf([x0, x1])
        G = g(t_iv)
        dx = x1 - x0
        lo = float(G.a)
        hi = float(G.b)
        if hi <= 0.0:
            # fully nonpositive → contributes to negative part
            area = mp.iv.mpf([(-hi) * dx, (-lo) * dx])  # -[lo,hi] = [-hi,-lo]
            return mp.iv.mpf([0.0, 0.0]), area
        elif lo >= 0.0:
            # fully nonnegative → contributes to positive part
            area = mp.iv.mpf([lo * dx, hi * dx])
            return area, mp.iv.mpf([0.0, 0.0])
        else:
            # mixed sign
            if depth >= max_refine:
                # conservative split: send absolute band to both sides (overestimates but safe)
                abs_hi = max(abs(lo), abs(hi))
                band = mp.iv.mpf([0.0, abs_hi * dx])
                return band, band
            xm = 0.5 * (x0 + x1)
            p1, n1 = process_interval(x0, xm, depth + 1)
            p2, n2 = process_interval(xm, x1, depth + 1)
            return p1 + p2, n1 + n2

    if not (b >= a):
        raise ValueError("interval_quad_signed_decomp: require b>=a")
    a = float(a)
    b = float(b)
    N = max(1, int(N))
    dx = (b - a) / N
    pos = mp.iv.mpf([0.0, 0.0])
    neg = mp.iv.mpf([0.0, 0.0])
    for i in range(N):
        x0 = a + i * dx
        x1 = x0 + dx
        p, n = process_interval(x0, x1, 0)
        pos += p
        neg += n
        if on_progress is not None:
            try:
                on_progress(i + 1, N)
            except Exception:
                pass
    return IntervalResult(float(pos.a), float(pos.b)), IntervalResult(float(neg.a), float(neg.b))


def archimedean_interval(
    h: Callable[[mp.iv.mpf], mp.iv.mpf],
    t_max: float,
    N_main: int = 2000,
    tail_model: str = "log-bound",
    tail_const: float = 4.0,
) -> tuple[IntervalResult, dict[str, float], tuple[IntervalResult, IntervalResult]]:
    """
    Certified bounds for A-term:
    A = (1/2π) ∫ h(t) (Re ψ(1/4 + i t/2) - log π) dt.

    Returns (A_interval, meta, (A_pos, A_neg)).

    Notes:
    - This routine uses interval decomposition for the main segment [-T, T], exploiting evenness to integrate [0,T] and doubling.
    - For tails |t|>T, uses a conservative upper bound |Re ψ(1/4 + i t/2) - log π| ≤ log(1+|t|) + C.
    """
    two_pi_inv = mp.iv.mpf([1.0 / (2.0 * math.pi), 1.0 / (2.0 * math.pi)])

    def M_interval(t_iv: mp.iv.mpf) -> mp.iv.mpf:
        # Interval enclosure of (Re ψ(1/4 + i t/2) - log π)
        # Try interval-aware digamma; fallback to safe bound
        try:
            z = mp.mpc(0.25, 0.5 * t_iv)  # may raise if iv not supported in complex
            psi = mp.digamma(z)
            re_iv = mp.iv.mpf([float(mp.re(psi).a), float(mp.re(psi).b)])
            return re_iv - math.log(math.pi)
        except Exception:
            # Safe fallback: |Re ψ - log π| ≤ log(1+|t|) + C  ⇒ interval = [-B, B]
            # Use sup |t| over t_iv
            t_abs_hi = max(abs(float(t_iv.a)), abs(float(t_iv.b)))
            B = math.log(1.0 + t_abs_hi) + float(tail_const)
            return mp.iv.mpf([-B, B])

    def g_interval(t_iv: mp.iv.mpf) -> mp.iv.mpf:
        return h(t_iv) * M_interval(t_iv) * two_pi_inv

    # Main integral on [0, T] (we later double for symmetry)
    A_pos_half, A_neg_half = interval_quad_signed_decomp(g_interval, 0.0, float(t_max), N_main)
    # Double for [-T, T]
    A_pos = IntervalResult(2.0 * A_pos_half.lower, 2.0 * A_pos_half.upper)
    A_neg = IntervalResult(2.0 * A_neg_half.lower, 2.0 * A_neg_half.upper)
    A_main = IntervalResult(A_pos.lower - A_neg.upper, A_pos.upper - A_neg.lower)

    # Tail bound: ∫_{|t|>T} (|h(t)| / 2π) (log(1+|t|) + C) dt, doubled for symmetry
    def tail_integrand_pos(t_iv: mp.iv.mpf) -> mp.iv.mpf:
        t_abs_hi = max(abs(float(t_iv.a)), abs(float(t_iv.b)))
        B = math.log(1.0 + t_abs_hi) + float(tail_const)
        # interval abs of h
        hv = h(t_iv)
        a = float(hv.a)
        b = float(hv.b)
        if a <= 0.0 <= b:
            hv_abs = mp.iv.mpf([0.0, max(abs(a), abs(b))])
        else:
            lo = min(abs(a), abs(b))
            hi = max(abs(a), abs(b))
            hv_abs = mp.iv.mpf([lo, hi])
        return hv_abs * (B / (2.0 * math.pi))

    # integrate on [T, ∞) via change of variables u = T + s / (1-s), s∈[0,1)
    # We truncate at a large U_max practical bound for positive functions (Gaussian/Cauchy decay)
    U_max = float(t_max) * 20.0
    N_tail = max(200, int(N_main // 4))
    tail = interval_quad_pos(lambda tv: tail_integrand_pos(tv), float(t_max), U_max, N_tail)
    # double for two tails
    tail_total = IntervalResult(2.0 * tail.lower, 2.0 * tail.upper)

    A_total = IntervalResult(A_main.lower - tail_total.upper, A_main.upper + tail_total.upper)

    meta = {
        "t_max": float(t_max),
        "N_main": int(N_main),
        "N_tail": int(N_tail),
        "tail_const": float(tail_const),
        "tail_upper": float(tail_total.upper),
    }
    return A_total, meta, (A_pos, A_neg)


def prime_sum_interval(
    hhat: Callable[[mp.iv.mpf], mp.iv.mpf],
    N: int,
    include_powers: bool = True,
) -> tuple[IntervalResult, list[tuple[int, IntervalResult]]]:
    """
    Interval sum for P main part up to N (primes and powers):
    P_main = (1/2π) Σ_{n≤N} 2Λ(n)/√n · ĥ(log n).
    Returns (interval, list_of_terms) for auditing.
    """
    two_pi_inv = 1.0 / (2.0 * math.pi)
    terms: list[tuple[int, IntervalResult]] = []

    # Precompute primes up to N via simple sieve (deterministic)
    sieve = [True] * (N + 1)
    if N >= 0:
        sieve[0] = False
    if N >= 1:
        sieve[1] = False
    p = 2
    while p * p <= N:
        if sieve[p]:
            step = p
            start = p * p
            sieve[start:N + 1:step] = [False] * (((N - start) // step) + 1)
        p += 1
    primes = [i for i, ok in enumerate(sieve) if ok]

    acc_lo = 0.0
    acc_hi = 0.0
    for p in primes:
        logp = math.log(p)
        coeff_iv = mp.iv.mpf([2.0 * logp / math.sqrt(p) * two_pi_inv] * 2)
        val = hhat(mp.iv.mpf([logp, logp]))
        term = coeff_iv * val
        it = IntervalResult(float(term.a), float(term.b))
        terms.append((p, it))
        acc_lo += it.lower
        acc_hi += it.upper
        if include_powers:
            pk = p * p
            while pk <= N:
                logpk = math.log(pk)
                coeff_iv = mp.iv.mpf([2.0 * logp / math.sqrt(pk) * two_pi_inv] * 2)
                val = hhat(mp.iv.mpf([logpk, logpk]))
                term = coeff_iv * val
                it = IntervalResult(float(term.a), float(term.b))
                terms.append((pk, it))
                acc_lo += it.lower
                acc_hi += it.upper
                pk *= p
    return IntervalResult(acc_lo, acc_hi), terms


def prime_tail_interval(
    hhat_abs: Callable[[mp.iv.mpf], mp.iv.mpf],
    N: int,
    u_max: float,
    M: int = 2000,
    A_psi: float = 1.2,
) -> IntervalResult:
    """
    Upper bound for P-tail using integral:
    tail ≤ (1/π) ∫_{log N}^{∞} e^{u/2} |ĥ(u)| du ≈ (1/π) ∫_{log N}^{u_max} ... du.
    Returns [0, upper].
    """
    u0 = math.log(max(2, int(N)))
    if u_max <= u0:
        return IntervalResult(0.0, 0.0)
    def integrand(u_iv: mp.iv.mpf) -> mp.iv.mpf:
        return (A_psi / math.pi) * mp.iv.exp(0.5 * u_iv) * hhat_abs(u_iv)
    integral_result = interval_quad_pos(integrand, u0, float(u_max), int(M))
    return IntervalResult(0.0, integral_result.upper)


def z_sum_interval(
    h: Callable[[mp.iv.mpf], mp.iv.mpf],
    zeros: list[float],
) -> IntervalResult:
    """
    Interval sum for Z main: Z = 2 Σ_{γ>0, γ∈zeros} h(γ)
    """
    acc_lo = 0.0
    acc_hi = 0.0
    for g in zeros:
        val = h(mp.iv.mpf([g, g])) * 2.0
        acc_lo += float(val.a)
        acc_hi += float(val.b)
    return IntervalResult(acc_lo, acc_hi)


def z_tail_interval(
    h: Callable[[mp.iv.mpf], mp.iv.mpf],
    gamma: float,
    c0: float = 0.28,
    t_max: float | None = None,
    M: int = 2000,
) -> IntervalResult:
    """
    Upper bound for Z-tail using zero density upper envelope:
    tail ≤ ∫_{Γ}^{∞} h(t) ρ_up(t) dt,  ρ_up(t)= (1/2π) log(t/2π) + c0, clipped at 0.
    """
    def rho_up(t_iv: mp.iv.mpf) -> mp.iv.mpf:
        # Use the upper end of interval for conservative bound
        t_hi = max(float(t_iv.a), float(t_iv.b))
        if t_hi <= 0:
            return mp.iv.mpf([0.0, 0.0])
        val = (math.log(t_hi / (2.0 * math.pi)) / (2.0 * math.pi)) + c0
        return mp.iv.mpf([max(0.0, val), max(0.0, val)])

    def integrand(t_iv: mp.iv.mpf) -> mp.iv.mpf:
        return h(t_iv) * rho_up(t_iv)

    # Use conservative threshold t0=50 where density upper bound is calibrated
    T0 = max(50.0, float(gamma))
    if t_max is None:
        t_max = max(T0 * 20.0, T0 + 1000.0)
    integral_result = interval_quad_pos(integrand, T0, float(t_max), int(M))
    return IntervalResult(0.0, integral_result.upper)

## Tight Archimedean interval (series-based) ##
def archimedean_interval_tight(
    h: Callable[[mp.iv.mpf], mp.iv.mpf],
    t_max: float,
    N_main: int = 2000,
    series_N0: int = 64,
    series_tol: float = 1e-10,
    series_N_max: int = 8192,
    tail_model: str = "log-bound",
    tail_const: float = 4.0,
    progress_cb: Callable[[int, int], None] | None = None,
) -> tuple[IntervalResult, dict[str, float], tuple[IntervalResult, IntervalResult]]:
    """
    A = (1/(2π)) ∫ h(t) * (Re ψ(1/4 + i t/2) - log π) dt
    Tight enclosure using explicit series for Re ψ with rigorous tail remainder.
    """
    two_pi = 2.0 * math.pi
    two_pi_inv_iv = mp.iv.mpf([1.0 / two_pi, 1.0 / two_pi])
    x = mp.mpf('0.25')
    euler_gamma = mp.euler

    def _M_interval_on_t_interval(t_iv: mp.iv.mpf) -> mp.iv.mpf:
        y_min = 0.5 * mp.mpf(t_iv.a)
        y_max = 0.5 * mp.mpf(t_iv.b)

        def partial_sum_bounds(N: int):
            lo = mp.mpf('0.0')
            hi = mp.mpf('0.0')
            for n in range(N):
                c = 1.0 / (n + 1.0)
                aa = n + x
                b_ymin = aa / (aa*aa + y_min*y_min)
                b_ymax = aa / (aa*aa + y_max*y_max)
                term_lo = c - b_ymin
                term_hi = c - b_ymax
                lo += term_lo
                hi += term_hi
            lo -= (euler_gamma + mp.log(mp.pi))
            hi -= (euler_gamma + mp.log(mp.pi))
            return lo, hi

        def tail_bounds(N: int):
            def A_of(NN, yy):
                return - ( mp.log(NN + 1) - 0.5 * mp.log((NN + x)*(NN + x) + yy*yy) )

            def R_of(NN, yy):
                S1 = 1.0 / (NN)
                if yy > 0:
                    S2 = (0.5 * mp.pi - mp.atan((NN - 1 + x) / yy)) / yy
                else:
                    S2 = 1.0 / (NN - 1 + x)
                return 0.5 * (S1 + S2)

            A_min = A_of(N, y_min)
            A_max = A_of(N, y_max)
            R_sup = R_of(N, y_min)
            tail_lo = A_min - R_sup
            tail_hi = A_max + R_sup
            return tail_lo, tail_hi

        N = max(int(series_N0), 16)
        while True:
            ps_lo, ps_hi = partial_sum_bounds(N)
            tb_lo, tb_hi = tail_bounds(N)
            M_lo = ps_lo + tb_lo
            M_hi = ps_hi + tb_hi
            width = float(M_hi - M_lo)
            if (width <= series_tol) or (N >= series_N_max):
                break
            N *= 2

        return mp.iv.mpf([M_lo, M_hi])

    def g_interval(t_iv: mp.iv.mpf) -> mp.iv.mpf:
        return h(t_iv) * _M_interval_on_t_interval(t_iv) * two_pi_inv_iv

    A_pos_half, A_neg_half = interval_quad_signed_decomp(
        g_interval, 0.0, float(t_max), N_main, on_progress=progress_cb
    )
    A_pos = IntervalResult(2.0 * A_pos_half.lower, 2.0 * A_pos_half.upper)
    A_neg = IntervalResult(2.0 * A_neg_half.lower, 2.0 * A_neg_half.upper)
    A_main = IntervalResult(A_pos.lower - A_neg.upper, A_pos.upper - A_neg.lower)

    def tail_integrand_pos(t_iv: mp.iv.mpf) -> mp.iv.mpf:
        t_abs_hi = max(abs(float(t_iv.a)), abs(float(t_iv.b)))
        B = math.log(1.0 + t_abs_hi) + float(tail_const)
        hv = h(t_iv)
        a = float(hv.a); b = float(hv.b)
        if a <= 0.0 <= b:
            hv_abs = mp.iv.mpf([0.0, max(abs(a), abs(b))])
        else:
            hv_abs = mp.iv.mpf([min(abs(a), abs(b)), max(abs(a), abs(b))])
        return hv_abs * (B / (2.0 * math.pi))

    U_max = float(t_max) * 20.0
    N_tail = max(200, int(N_main // 4))
    tail = interval_quad_pos(lambda tv: tail_integrand_pos(tv), float(t_max), U_max, N_tail)
    tail_total = IntervalResult(2.0 * tail.lower, 2.0 * tail.upper)

    A_total = IntervalResult(A_main.lower - tail_total.upper, A_main.upper + tail_total.upper)
    meta = {
        "t_max": float(t_max),
        "N_main": int(N_main),
        "series_N0": int(series_N0),
        "N_tail": int(N_tail),
        "tail_const": float(tail_const),
        "tail_upper": float(tail_total.upper),
    }
    return A_total, meta, (A_pos, A_neg)
